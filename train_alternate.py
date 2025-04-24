import time
import torch

from tqdm import trange
from models.kplane_field import init_grid_param
from models.model_io import save_models
from utils.regularizers import get_regularizers
from ops.schedulers import get_cosine_schedule_with_warmup
from ops.appearance_optimizers import optimize_appearance_codes
from utils.metrics import psnr

def train_alternate(
        config,
        device,
        l_epoch,
        tr_loader,
        noisy_latent_code,
        k_planes_diffuser,
        kplanes_model,
        writer,
        ts_dset=None,
        optimizer_state_dicts=None):
    if (config['dataset_name'] == 'nerf_synthetic'):
        train_alternate_synth(
            config,
            device,
            l_epoch,
            tr_loader,
            noisy_latent_code,
            k_planes_diffuser,
            kplanes_model,
            writer,
            optimizer_state_dicts)
    elif (config['dataset_name'] == 'phototourism'):
        train_alternate_phototourism(
            config,
            device,
            l_epoch,
            tr_loader,
            noisy_latent_code,
            k_planes_diffuser,
            kplanes_model,
            writer,
            ts_dset,
            optimizer_state_dicts)
    else:
        raise NotImplementedError(
            f"Alternate training for {config['dataset_name']} not yet implemented.")


def train_alternate_phototourism(
        config,
        device,
        l_epoch,
        tr_loader,
        noisy_latent_code,
        k_planes_diffuser,
        kplanes_model,
        writer,
        ts_dset,
        optimizer_state_dicts=None):
    if (l_epoch + 1 == config['epochs']):
        print(f"All {config['epochs']} epochs already done.")
        return
    grid_config = config['grid_config']

    # Parameters to optimize
    # unet attn procs
    lora_params = k_planes_diffuser.lora_layers.parameters()
    # Decoder params
    decoder_params = k_planes_diffuser.vae.decoder.parameters()
    # tiny-MLP params
    kplanes_model_params = kplanes_model.get_params(lr=config['lr_kplanes'])

    # Defining trainable parameters for the optimizer
    kplanes_trainable_params = kplanes_model_params
    diffusion_trainable_params = list(lora_params)
    diffusion_trainable_params += list(decoder_params)


    # Defining optimizers
    kplanes_optimizer = torch.optim.Adam(
        kplanes_trainable_params,
        lr=config['lr_kplanes'])
    if (optimizer_state_dicts is not None):
        kplanes_optimizer.load_state_dict(optimizer_state_dicts[0])
    diffusion_optimizer = torch.optim.Adam(
        diffusion_trainable_params,
        lr=config['lr_diffusion'])
    if (optimizer_state_dicts is not None):
        diffusion_optimizer.load_state_dict(optimizer_state_dicts[1])

    # Defining criterions
    kplanes_criterion = torch.nn.MSELoss(reduction='mean')
    diffusion_criterion = torch.nn.MSELoss(reduction='mean')

    # Defining lr schedulers
    kplanes_lr_scheduler = get_cosine_schedule_with_warmup(
        kplanes_optimizer,
        num_warmup_steps=config['kplanes_lr_warmup_steps'],
        num_training_steps=len(tr_loader) *
        config['epochs_kplanes'])

    # gscalers for fp16 training
    kplanes_gscaler = torch.cuda.amp.GradScaler(enabled=config['train_fp16'])

    # Defining regularizers
    kplanes_regularizers = [r for r in get_regularizers(config)]

    it_idx = 0
    print(f"Training models for {config['epochs']-l_epoch} epoch(s)...")
    print(f"\tepochs_kplanes: {config['epochs_kplanes']}")
    print(f"\tepochs_diffusion: {config['epochs_diffusion']}")

    it_idx = 0
    k_planes_diffuser.lora_layers.train()
    k_planes_diffuser.vae.train()
    compile_k_planes_diffuser = config['compile_k_planes_diffuser']
    kpd = torch.compile(k_planes_diffuser, mode="reduce-overhead") if compile_k_planes_diffuser else k_planes_diffuser

    for epoch in trange(l_epoch + 1, config['epochs'] + 1):
        kplanes_running_loss = 0
        kplanes_running_psnr = 0
        diffusion_loss = 0

        # Reinitializing the kplanes optimizer and lr scheduler
        if (not config['keep_scheduler']):
            if (epoch > 1):
                kplanes_trainable_params = kplanes_model.get_params(
                    lr=config['lr_kplanes_finetune'])
                kplanes_optimizer = torch.optim.Adam(
                    kplanes_trainable_params, lr=config['lr_kplanes_finetune'])
                kplanes_lr_scheduler = get_cosine_schedule_with_warmup(
                    kplanes_optimizer,
                    num_warmup_steps=config['kplanes_lr_warmup_steps'],
                    num_training_steps=len(tr_loader) *
                    config['epochs_kplanes'])

        for epoch_kplanes in trange(config['epochs_kplanes']):
            kplanes_running_loss = 0
            kplanes_running_psnr = 0
            for i, e in enumerate(tr_loader):
                kplanes_model.step_before_iter(it_idx)
                kplanes_optimizer.zero_grad()
                kplanes_model.train()

                e['rays_o'] = e['rays_o'].to(device)
                e['rays_d'] = e['rays_d'].to(device)
                e['bg_color'] = e['bg_color'].to(device)
                e['near_fars'] = e['near_fars'].to(device)
                e['imgs'] = e['imgs'].to(device)
                if "timestamps" in e:
                    e["timestamps"] = e["timestamps"].to(device)
                else:
                    e["timestamps"] = None

                with torch.cuda.amp.autocast(enabled=config['train_fp16']):
                    fwd_out = kplanes_model(e['rays_o'],
                                            e['rays_d'],
                                            bg_color=e['bg_color'],
                                            near_far=e['near_fars'],
                                            timestamps=e['timestamps'])
                    recon_loss = kplanes_criterion(fwd_out['rgb'], e['imgs'])
                    kplanes_loss = recon_loss

                    for r in kplanes_regularizers:
                        reg_loss = r.regularize(
                            kplanes_model, model_out=fwd_out)
                        kplanes_loss = kplanes_loss + reg_loss

                current_psnr = psnr(fwd_out['rgb'], e['imgs'])
                kplanes_running_loss += kplanes_loss.item()
                kplanes_running_psnr += current_psnr

                writer.add_scalar(
                    "kplanes_metrics/train/kplanes_it_loss",
                    kplanes_loss.item(),
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/kplanes_it_psnr",
                    current_psnr,
                    it_idx)
                
                kplanes_gscaler.scale(kplanes_loss).backward()
                kplanes_gscaler.step(kplanes_optimizer)
                kplanes_scale = kplanes_gscaler.get_scale()
                kplanes_gscaler.update()
                kplanes_lr_scheduler.step()

                writer.add_scalar(
                    "kplanes_metrics/train/len_lr",
                    len(kplanes_trainable_params),
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/lr_sched_0",
                    kplanes_lr_scheduler.get_last_lr()[0],
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/lr_sched_1",
                    kplanes_lr_scheduler.get_last_lr()[1],
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/lr_sched_2",
                    kplanes_lr_scheduler.get_last_lr()[2],
                    it_idx)

                for r in kplanes_regularizers:
                    r.step(it_idx)

                kplanes_model.step_after_iter(it_idx)

                del e['rays_o'], e['rays_d'], e['bg_color'], e['near_fars'], e['imgs'], e['timestamps'], fwd_out
                del kplanes_loss, recon_loss, reg_loss

                if (i % config['kplanes_print_every'] == 0 and i > 0):
                    print(f"\tepoch: {epoch}\tepoch_kplanes: {epoch_kplanes}\ti: {i}\tkplanes_running_loss: {kplanes_running_loss/config['kplanes_print_every']}")
                    print(f"kplanes_running_psnr: {kplanes_running_psnr/config['kplanes_print_every']}")
                    kplanes_running_loss = 0
                    kplanes_running_psnr = 0

                if (i % config['valid_every'] == 0):
                    app_optim_settings = {
                        'app_optim_epochs': config['app_optim_epochs'],
                        'app_optim_lr': config['app_optim_lr'],
                        'app_optim_batch_size': config['app_optim_batch_size']
                    }
                    optimize_appearance_codes(
                        kplanes_model,
                        ts_dset,
                        device,
                        config['train_fp16'],
                        kplanes_criterion,
                        writer,
                        it_idx,
                        app_optim_settings)

                it_idx += 1

            del current_psnr, kplanes_running_loss, kplanes_running_psnr

        if (epoch % config['save_every'] == 0 and epoch > 1):
            print(f"Saving checkpoint_{epoch}...")
            save_models(config, epoch, noisy_latent_code, k_planes_diffuser, kplanes_model, [
                        kplanes_optimizer, diffusion_optimizer], [kplanes_lr_scheduler], diffusion_loss)

        if (config['epochs_diffusion'] > 0):
            target_grids = kplanes_model.field.grids[0]
            target_grids = torch.cat(
                (target_grids[0], target_grids[1], target_grids[2]), axis=0)
            target_grids = target_grids.view(
                1,
                target_grids.shape[0],
                target_grids.shape[1],
                target_grids.shape[2],
                target_grids.shape[3])
            target_grids = target_grids.detach()

            for epoch_diffusion in range(config['epochs_diffusion']):
                diffusion_optimizer.zero_grad()
                kplanes_model.eval()

                # Generating K-Planes with KPlanesDiffuser
                kplanes = kpd(noisy_latent_code, config['prompt'])

                # Reshaping K-Planes
                kplanes = kplanes.view([kplanes.shape[0],
                                        config['nb_kplanes'],
                                        config['kplanes_channel_dim'],
                                        config['kplanes_resolution'],
                                        config['kplanes_resolution']]).to(device)

                diffusion_loss = diffusion_criterion(kplanes, target_grids)

                writer.add_scalar("diffusion_metrics/train/loss",
                                  diffusion_loss.item(),
                                  (config['epochs_diffusion'] * (epoch - 1)) + epoch_diffusion)

                if (epoch_diffusion % config['diffusion_print_every'] == 0):
                    print(f"\tepoch_diffusion: {epoch_diffusion}, diffusion_loss: {diffusion_loss.item()}")

                diffusion_loss.backward()
                diffusion_optimizer.step()

            # Setting K-Planes in low-rank model
            gp = init_grid_param(
                grid_nd=grid_config[0]["grid_dimensions"],
                in_dim=grid_config[0]["input_coordinate_dim"],
                out_dim=grid_config[0]["output_coordinate_dim"],
                reso=grid_config[0]["resolution"],
                init_kplanes=kplanes
            )
            kplanes_model.field.grids[0] = gp



def train_alternate_synth(
        config,
        device,
        l_epoch,
        tr_loader,
        noisy_latent_code,
        k_planes_diffuser,
        kplanes_model,
        writer,
        optimizer_state_dicts=None):
    if (l_epoch + 1 == config['epochs']):
        print(f"All {config['epochs']} epochs already done.")
        return
    grid_config = config['grid_config']

    # Parameters to optimize
    # unet attn procs
    lora_params = k_planes_diffuser.lora_layers.parameters()
    # Decoder params
    decoder_params = k_planes_diffuser.vae.decoder.parameters()
    # tiny-MLP params
    kplanes_model_params = kplanes_model.get_params(lr=config['lr_kplanes'])

    # Defining trainable parameters for the optimizer
    kplanes_trainable_params = kplanes_model_params
    diffusion_trainable_params = list(lora_params)
    diffusion_trainable_params += list(decoder_params)

    # Defining optimizers
    kplanes_optimizer = torch.optim.Adam(
        kplanes_trainable_params,
        lr=config['lr_kplanes'])
    if (optimizer_state_dicts is not None):
        kplanes_optimizer.load_state_dict(optimizer_state_dicts[0])
    diffusion_optimizer = torch.optim.Adam(
        diffusion_trainable_params,
        lr=config['lr_diffusion'])
    if (optimizer_state_dicts is not None):
        diffusion_optimizer.load_state_dict(optimizer_state_dicts[1])

    # Defining criterions
    kplanes_criterion = torch.nn.MSELoss(reduction='mean')
    diffusion_criterion = torch.nn.MSELoss(reduction='mean')

    # Defining lr schedulers
    kplanes_lr_scheduler = get_cosine_schedule_with_warmup(
        kplanes_optimizer,
        num_warmup_steps=config['kplanes_lr_warmup_steps'],
        num_training_steps=len(tr_loader) *
        config['epochs_kplanes'])

    # gscalers for fp16 training
    kplanes_gscaler = torch.cuda.amp.GradScaler(enabled=config['train_fp16'])

    # Defining regularizers
    kplanes_regularizers = [r for r in get_regularizers(config)]

    it_idx = 0
    print(f"Alternate training models for {config['epochs']-l_epoch} epoch(s)...")
    print(f"\tepochs_kplanes: {config['epochs_kplanes']}")
    print(f"\tepochs_diffusion: {config['epochs_diffusion']}")

    it_idx = 0
    for epoch in trange(l_epoch + 1, config['epochs'] + 1):
        kplanes_running_loss = 0
        kplanes_running_psnr = 0
        diffusion_loss = 0

        # Reinitializing the kplanes optimizer and lr scheduler
        if (not config['keep_scheduler']):
            if (epoch > 1):
                kplanes_trainable_params = kplanes_model.get_params(
                    lr=config['lr_kplanes_finetune'])
                kplanes_optimizer = torch.optim.Adam(
                    kplanes_trainable_params, lr=config['lr_kplanes_finetune'])
                kplanes_lr_scheduler = get_cosine_schedule_with_warmup(
                    kplanes_optimizer,
                    num_warmup_steps=config['kplanes_lr_warmup_steps'],
                    num_training_steps=len(tr_loader) *
                    config['epochs_kplanes'])

        for epoch_kplanes in trange(config['epochs_kplanes']):
            kplanes_running_loss = 0
            kplanes_running_psnr = 0
            for i, e in enumerate(tr_loader):
                kplanes_model.step_before_iter(it_idx)
                kplanes_optimizer.zero_grad()
                k_planes_diffuser.lora_layers.eval()
                k_planes_diffuser.vae.eval()
                kplanes_model.train()

                # Dropping last batch
                if (e['rays_o'].shape[0] < config['batch_size']):
                    continue

                e['rays_o'] = e['rays_o'].to(device)
                e['rays_d'] = e['rays_d'].to(device)
                e['bg_color'] = e['bg_color'].to(device)
                e['near_fars'] = e['near_fars'].to(device)
                e['imgs'] = e['imgs'].to(device)
                if "timestamps" in e:
                    e["timestamps"] = e["timestamps"].to(device)
                else:
                    e["timestamps"] = None

                with torch.cuda.amp.autocast(enabled=config['train_fp16']):
                    fwd_out = kplanes_model(e['rays_o'],
                                            e['rays_d'],
                                            bg_color=e['bg_color'],
                                            near_far=e['near_fars'],
                                            timestamps=e['timestamps'])
                    recon_loss = kplanes_criterion(fwd_out['rgb'], e['imgs'])
                    kplanes_loss = recon_loss

                    for r in kplanes_regularizers:
                        reg_loss = r.regularize(
                            kplanes_model, model_out=fwd_out)
                        kplanes_loss = kplanes_loss + reg_loss

                current_psnr = psnr(fwd_out['rgb'], e['imgs'])
                kplanes_running_loss += kplanes_loss.item()
                kplanes_running_psnr += current_psnr

                writer.add_scalar(
                    "kplanes_metrics/train/kplanes_it_loss",
                    kplanes_loss.item(),
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/kplanes_it_psnr",
                    current_psnr,
                    it_idx)
                
                kplanes_gscaler.scale(kplanes_loss).backward()
                kplanes_gscaler.step(kplanes_optimizer)
                kplanes_scale = kplanes_gscaler.get_scale()
                kplanes_gscaler.update()
                kplanes_lr_scheduler.step()

                writer.add_scalar(
                    "kplanes_metrics/train/len_lr",
                    len(kplanes_trainable_params),
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/lr_sched_0",
                    kplanes_lr_scheduler.get_last_lr()[0],
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/lr_sched_1",
                    kplanes_lr_scheduler.get_last_lr()[1],
                    it_idx)
                writer.add_scalar(
                    "kplanes_metrics/train/lr_sched_2",
                    kplanes_lr_scheduler.get_last_lr()[2],
                    it_idx)

                for r in kplanes_regularizers:
                    r.step(it_idx)

                kplanes_model.step_after_iter(it_idx)

                del e['rays_o'], e['rays_d'], e['bg_color'], e['near_fars'], e['imgs'], e['timestamps'], fwd_out
                del kplanes_loss, recon_loss, reg_loss

                if (i % config['kplanes_print_every'] == 0 and i > 0):
                    print(f"\tepoch: {epoch}\tepoch_kplanes: {epoch_kplanes}\ti: {i}\tkplanes_running_loss: {kplanes_running_loss/config['kplanes_print_every']}")
                    print(f"kplanes_running_psnr: {kplanes_running_psnr/config['kplanes_print_every']}")
                    kplanes_running_loss = 0
                    kplanes_running_psnr = 0

                it_idx += 1

            del current_psnr, kplanes_running_loss, kplanes_running_psnr

        if (epoch % config['save_every'] == 0):
            print(f"Saving checkpoint_{epoch}...")
            save_models(config, epoch, noisy_latent_code, k_planes_diffuser, kplanes_model, [
                        kplanes_optimizer, diffusion_optimizer], [kplanes_lr_scheduler], diffusion_loss)

        if (config['epochs_diffusion'] > 0):
            target_grids = kplanes_model.field.grids[0]
            target_grids = torch.cat(
                (target_grids[0], target_grids[1], target_grids[2]), axis=0)
            target_grids = target_grids.view(
                1,
                target_grids.shape[0],
                target_grids.shape[1],
                target_grids.shape[2],
                target_grids.shape[3])
            target_grids = target_grids.detach()

            for epoch_diffusion in range(config['epochs_diffusion']):
                diffusion_optimizer.zero_grad()

                k_planes_diffuser.lora_layers.train()
                k_planes_diffuser.vae.train()
                kplanes_model.eval()

                # Generating K-Planes with KPlanesDiffuser
                kplanes = k_planes_diffuser(noisy_latent_code, config['prompt'])

                # Reshaping K-Planes
                kplanes = kplanes.view([kplanes.shape[0],
                                        config['nb_kplanes'],
                                        config['kplanes_channel_dim'],
                                        config['kplanes_resolution'],
                                        config['kplanes_resolution']]).to(device)

                diffusion_loss = diffusion_criterion(kplanes, target_grids)

                writer.add_scalar("diffusion_metrics/train/loss",
                                  diffusion_loss.item(),
                                  (config['epochs_diffusion'] * (epoch - 1)) + epoch_diffusion)

                if (epoch_diffusion % config['diffusion_print_every'] == 0):
                    print(f"\tepoch_diffusion: {epoch_diffusion}, diffusion_loss: {diffusion_loss.item()}")

                diffusion_loss.backward()
                diffusion_optimizer.step()

            # Setting K-Planes in low-rank model
            gp = init_grid_param(
                grid_nd=grid_config[0]["grid_dimensions"],
                in_dim=grid_config[0]["input_coordinate_dim"],
                out_dim=grid_config[0]["output_coordinate_dim"],
                reso=grid_config[0]["resolution"],
                init_kplanes=kplanes
            )
            kplanes_model.field.grids[0] = gp
