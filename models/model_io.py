import glob
import os
import torch

from models.KPlanesDiffuser import KPlanesDiffuser
from models.lowrank_model import LowrankModel

def init_models(device, config, grid_config, tr_dset, extra_args, create_attn_procs = True, ts_dset=None):
    # Defining a KPlanes Diffuser
    k_planes_diffuser = KPlanesDiffuser(pretrained_model_name_or_path=config['pretrained_model_name_or_path'],
                                        revision=config['revision'],
                                        output_dir=config['output_dir'],
                                        cache_dir=config['cache_dir'],
                                        logging_dir=config['log_dir'],
                                        latent_resolution=config['latent_resolution'],
                                        latent_channel_dim=config['latent_channel_dim'],
                                        kplanes_resolution=config['kplanes_resolution'],
                                        kplanes_channel_dim=config['kplanes_channel_dim'],
                                        enable_xformers_memory_efficient_attention=config['enable_xformers_memory_efficient_attention'],
                                        create_attn_procs=create_attn_procs,
                                        device=device).to(device)
    
    # Sampling a gaussian latent noise
    noisy_latent_code = torch.randn(config['noise_batch_size'], 
                                    config['latent_channel_dim'],
                                    config['latent_resolution'], 
                                    config['latent_resolution']).to(device)
    
    # Sampling latent K-Planes and decoding
    with torch.no_grad():
        # latent_kplanes = k_planes_diffuser(noisy_latent_code, config['prompt'])
        # latent_kplanes = 1 / 0.18215 * latent_kplanes
        # kplanes = k_planes_diffuser.vae.decode(latent_kplanes).sample
        kplanes = k_planes_diffuser(noisy_latent_code, config['prompt'])

    # Reshaping
    # The shape is now (nb_batches, nb_kplanes, kplanes_channel_dim, kplanes_resolution, kplanes_resolution)
    kplanes = kplanes.view([kplanes.shape[0],
                    config['nb_kplanes'], 
                    config['kplanes_channel_dim'], 
                    config['kplanes_resolution'], 
                    config['kplanes_resolution']]).to(device)

    
    # K-Planes dataset management
    try:
        global_scale = tr_dset.global_scale
    except AttributeError:
        global_scale = None
    try:
        global_translation = tr_dset.global_translation
    except AttributeError:
        global_translation = None
    try:
        num_images = tr_dset.num_images
    except AttributeError:
        num_images = None

    # Initializing K-Planes LowrankModel with the output
    kplanes_model = LowrankModel(grid_config=grid_config,
                                aabb=tr_dset.scene_bbox,
                                is_ndc=tr_dset.is_ndc,
                                is_contracted=tr_dset.is_contracted,
                                global_scale=global_scale,
                                global_translation=global_translation,
                                use_appearance_embedding=config['use_appearance_embeddings'],
                                appearance_embedding_dim=config['appearance_embeddings_dim'],
                                num_images=num_images,
                                init_kplanes=kplanes,
                                linear_decoder=config['explicit_decoding'],
                                **extra_args).to(device)
    
    # Initializing appearance embeddings
    if(config['load_checkpoint'] and config['use_appearance_embeddings']):
        # 1. Initialize test appearance code to average code.
        num_test_imgs = len(ts_dset)
        if not hasattr(kplanes_model.field, "test_appearance_embedding"):
            tst_embedding = torch.nn.Embedding(
                num_test_imgs, kplanes_model.field.appearance_embedding_dim
            ).to(device)
            with torch.autograd.no_grad():
                tst_embedding.weight.copy_(
                    kplanes_model.field.appearance_embedding.weight
                        .detach()
                        .mean(dim=0, keepdim=True)
                        .expand(num_test_imgs, -1)
                )
            kplanes_model.field.test_appearance_embedding = tst_embedding
    
    if (config['verbose']):
        print(f"\nnoisy_latent_code.shape: {noisy_latent_code.shape}")
        print(f"kplanes.shape: {kplanes.shape}\n")

    del kplanes
    
    return [k_planes_diffuser, kplanes_model, noisy_latent_code]

def save_models(config, epoch, noisy_latent_code, k_planes_diffuser, kplanes_model, optimizer, lr_scheduler, loss):
    target_dir = f"{config['checkpoint_dir']}/checkpoint_{epoch}/checkpoint_{epoch}.pt"
    procs_target_dir = f"{config['checkpoint_dir']}/checkpoint_{epoch}"
    k_planes_diffuser.unet = k_planes_diffuser.unet.to(torch.float32)

    k_planes_diffuser.unet.save_attn_procs(procs_target_dir)
        
    loss_item = loss.item() if hasattr(loss, 'item') else loss
    torch.save({
        'epoch': epoch,
        'noisy_latent_code': noisy_latent_code,
        'k_planes_diffuser_vae_state_dict': k_planes_diffuser.vae.state_dict(),
        'kplanes_model_state_dict': kplanes_model.state_dict(),
        'optimizer_state_dict': [opt.state_dict() for opt in optimizer],
        'lr_scheduler_state_dict': [lrs.state_dict() for lrs in lr_scheduler],
        'loss': loss_item,
    }, target_dir)

def load_models(config, k_planes_diffuser, kplanes_model, checkpoint=-1):
    if (checkpoint == -1):
        checkpoints = sorted(glob.glob(config['checkpoint_dir'] + '/*'))
        if (len(checkpoints)==0):
            raise Exception(f"Tried to load checkpoint from {config['checkpoint_dir']}/ but 0 found.")
        checkpoint_path = checkpoints[checkpoint]
        checkpoint_filename = checkpoint_path.split('/')[-1]
    else:
        checkpoint_path = f"{config['checkpoint_dir']}/checkpoint_{checkpoint}"
        checkpoint_filename = f"checkpoint_{checkpoint}.pt"
    saved_checkpoint = torch.load(f"{checkpoint_path}/{checkpoint_filename}")

    epoch = saved_checkpoint['epoch']
    noisy_latent_code = saved_checkpoint['noisy_latent_code']
    k_planes_diffuser_vae_state_dict = saved_checkpoint['k_planes_diffuser_vae_state_dict']
    kplanes_model_state_dict = saved_checkpoint['kplanes_model_state_dict']
    optimizer_state_dict = saved_checkpoint['optimizer_state_dict']
    lr_scheduler_state_dict = saved_checkpoint['lr_scheduler_state_dict']
    loss = saved_checkpoint['loss']

    k_planes_diffuser.load_checkpoint_attn_procs(checkpoint_path)
    k_planes_diffuser.vae.load_state_dict(k_planes_diffuser_vae_state_dict)
    kplanes_model.load_state_dict(kplanes_model_state_dict)

    print(f"Loaded model from {checkpoint_path}.")
    del k_planes_diffuser_vae_state_dict, kplanes_model_state_dict

    return epoch, noisy_latent_code, k_planes_diffuser, kplanes_model, optimizer_state_dict, lr_scheduler_state_dict, loss