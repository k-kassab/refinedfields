import math
import torch
from tqdm import tqdm 

def optimize_appearance_codes(kplanes_model, dset, device, train_fp16, criterion, writer, it_idx, app_optim_settings):
    """Optimize the appearance embedding of all test poses.
        """
    
    num_test_imgs = len(dset)

    # 1. Initialize test appearance code to average code.
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

    # 2. Setup parameter trainability
    kplanes_model.eval()
    param_trainable = {}
    for pn, p in kplanes_model.named_parameters():
        param_trainable[pn] = p.requires_grad
        p.requires_grad_(False)
    kplanes_model.field.test_appearance_embedding.requires_grad_(True)

    # 3. Optimize
    pb = tqdm(total=len(dset), desc=f"Appearance-code optimization")
    for img_idx, data in enumerate(dset):
        optimize_appearance_step(kplanes_model, data, img_idx, device, train_fp16, criterion, writer, it_idx, app_optim_settings)
        pb.update(1)
    pb.close()

    # 4. Reset parameter trainability
    for pn, p in kplanes_model.named_parameters():
        p.requires_grad_(param_trainable[pn])
    kplanes_model.field.test_appearance_embedding.requires_grad_(False)

def optimize_appearance_step(kplanes_model, data, im_id, device, train_fp16, criterion, writer, it_idx, app_optim_settings):
    rays_o = data["rays_o_left"]
    rays_d = data["rays_d_left"]
    imgs = data["imgs_left"]
    near_far = data["near_fars"]
    bg_color = data["bg_color"]
    if isinstance(bg_color, torch.Tensor):
        bg_color = bg_color.to(device)

    epochs = app_optim_settings['app_optim_epochs']
    batch_size = app_optim_settings['app_optim_batch_size']
    app_optim_lr = app_optim_settings['app_optim_lr']
    n_steps = math.ceil(rays_o.shape[0] / batch_size)

    camera_id = torch.full((batch_size, ), fill_value=im_id, dtype=torch.int32, device=device)

    app_optim = torch.optim.Adam(params=kplanes_model.field.test_appearance_embedding.parameters(), lr=app_optim_lr)
    lr_sched = torch.optim.lr_scheduler.StepLR(app_optim, step_size=3 * n_steps, gamma=0.1)
    lowest_loss, lowest_loss_count = 100_000_000, 0
    grad_scaler = torch.cuda.amp.GradScaler(enabled=train_fp16)
    for n in range(epochs):
        idx = torch.randperm(rays_o.shape[0])
        for b in range(n_steps):
            batch_ids = idx[b * batch_size: (b + 1) * batch_size]
            rays_o_b = rays_o[batch_ids].to(device)
            rays_d_b = rays_d[batch_ids].to(device)
            imgs_b = imgs[batch_ids].to(device)
            near_far_b = near_far[batch_ids].to(device)
            camera_id_b = camera_id[:len(batch_ids)]

            with torch.cuda.amp.autocast(enabled=train_fp16):
                fwd_out = kplanes_model(
                    rays_o_b, rays_d_b, timestamps=camera_id_b, bg_color=bg_color,
                    near_far=near_far_b)
                recon_loss = criterion(fwd_out['rgb'], imgs_b)
            app_optim.zero_grad(set_to_none=True)
            grad_scaler.scale(recon_loss).backward()
            grad_scaler.step(app_optim)
            grad_scaler.update()
            lr_sched.step()

            writer.add_scalar(
                f"appearance_loss_{it_idx}/recon_loss_{im_id}", recon_loss.item(),
                b + n * n_steps)

            if recon_loss.item() < lowest_loss:
                lowest_loss = recon_loss.item()
                lowest_loss_count = 0
            lowest_loss_count += 1
        # 1 epoch without improvement -> stop
        if lowest_loss_count > 1 * n_steps:
            break