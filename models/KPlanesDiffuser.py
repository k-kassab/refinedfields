from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from packaging import version
from accelerate.logging import get_logger
from transformers import CLIPTextModel, CLIPTokenizer
from utils.utils import modify_layer
import torch

logger = get_logger(__name__, log_level="INFO")


class KPlanesDiffuser(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 revision: str = None,
                 output_dir: str = "outputs/",
                 cache_dir: str = "cache/",
                 logging_dir: str = "logs/",
                 latent_resolution: int = 64,
                 latent_channel_dim: int = 4,
                 nb_kplanes: int = 3,
                 kplanes_resolution: int = 512,
                 kplanes_channel_dim: int = 32,
                 enable_xformers_memory_efficient_attention: bool = True,
                 device: torch.device = torch.device("cpu"),
                 create_attn_procs: bool = True):
        super(KPlanesDiffuser, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.revision = revision
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.logging_dir = logging_dir
        self.latent_resolution = latent_resolution
        self.latent_channel_dim = latent_channel_dim
        self.nb_kplanes = nb_kplanes
        self.kplanes_resolution = kplanes_resolution
        self.kplanes_channel_dim = kplanes_channel_dim
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
        self.device = device
        self.create_attn_procs = create_attn_procs

        # Importing pretrained models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet", revision=self.revision)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.revision)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision)

        # Freezing unet
        self.unet.requires_grad_(False)

        # Moving models to device
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)

        # VAE modification
        # Modifying last decoder layer
        self.vae._modules['decoder']._modules['conv_out'] = modify_layer(
            self.vae._modules['decoder']._modules['conv_out'],
            out_channels=self.nb_kplanes * self.kplanes_channel_dim
        )

        # Modify the latent space dimension if it's not set to default
        if (self.latent_channel_dim != 4):
            self.unet._modules['conv_in'] = modify_layer(
                self.unet._modules['conv_in'],
                in_channels=self.latent_channel_dim
            )
            self.unet._modules['conv_out'] = modify_layer(
                self.unet._modules['conv_out'],
                out_channels=self.latent_channel_dim
            )
            self.vae._modules['decoder']._modules['conv_in'] = modify_layer(
                self.vae._modules['decoder']._modules['conv_in'],
                in_channels=self.latent_channel_dim
            )
            self.vae._modules['post_quant_conv'] = modify_layer(
                self.vae._modules['post_quant_conv'],
                in_channels=self.latent_channel_dim,
                out_channels=self.latent_channel_dim
            )

        # Setting LoRA weights for the unet
        self.lora_layers = []
        if (self.create_attn_procs):
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith(
                    "attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(
                        reversed(
                            self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

            # Checking xformers
            if self.enable_xformers_memory_efficient_attention:
                if is_xformers_available():
                    import xformers

                    xformers_version = version.parse(xformers.__version__)
                    if xformers_version == version.parse("0.0.16"):
                        logger.warn(
                            "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                        )
                    self.unet.enable_xformers_memory_efficient_attention()
                else:
                    raise ValueError(
                        "xformers is not available. Make sure it is installed correctly")

            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

    def forward(self, x, prompt: str):
        # Make sure x has the expected resolution and number of channels
        assert x.shape[1:] == torch.Size(
            [self.latent_channel_dim, self.latent_resolution, self.latent_resolution])

        # Conditioning on the input prompt
        text_input = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device))[0]
            text_embeddings = torch.repeat_interleave(
                text_embeddings, x.shape[0], dim=0)

        timestep = torch.tensor(
            self.noise_scheduler.config.num_train_timesteps).to(
            self.device)
        timestep = torch.repeat_interleave(timestep, x.shape[0])

        latent_kplanes = self.unet(x, timestep, text_embeddings).sample
        latent_kplanes = 1 / 0.18215 * latent_kplanes
        kplanes = self.vae.decode(latent_kplanes).sample
        
        return kplanes

    def load_checkpoint_attn_procs(self, checkpoint_path):
        self.unet.load_attn_procs(checkpoint_path)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

        # Checking xformers
        if self.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")
        