import torch

from tqdm import tqdm
from utils.utils import save_image_from_tensor
from utils.metrics import psnr, ssim
import lpips


def test(config, device, kplanes_model, ts_dset, epoch_nb=None):
    psnrs = {}
    ssims = {}
    mses = {}
    lpipss = {}
    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    total_lpips = 0

    kplanes_model.eval()

    print("\nTesting models...")
    pb = tqdm(total=len(ts_dset))
    torch.set_printoptions(precision=8)
    eval_renders = True
    with torch.no_grad():
        bs = config['batch_size']
        lpips_loss = lpips.LPIPS(net='alex').to(device)
        for i, e in enumerate(ts_dset):
            # GT image
            img_gt = e['imgs'].view(e['img_h'], e['img_w'], 3)
            img_gt = img_gt * 255.0

            # Rendering images from model
            fwd_out_rgb = torch.tensor([])
            for sub_e_idx in range(0, e['rays_o'].shape[0] - bs, bs):
                sub_e = {}
                sub_e['rays_o'] = e['rays_o'][sub_e_idx:sub_e_idx+bs].to(device)
                sub_e['rays_d'] = e['rays_d'][sub_e_idx:sub_e_idx+bs].to(device)
                sub_e['bg_color'] = e['bg_color'].repeat(bs, 1).to(device)
                if (config['dataset_name'] == 'nerf_synthetic'):
                    sub_e['near_fars'] = e['near_fars'].repeat(bs, 1).to(device)
                    sub_e['timestamps'] = None
                else:
                    sub_e['near_fars'] = e['near_fars'][sub_e_idx:sub_e_idx+bs].to(device)
                    sub_e['timestamps'] = e['timestamps'][sub_e_idx:sub_e_idx+bs].to(device)

                sub_fwd_out = kplanes_model(sub_e['rays_o'],
                                            sub_e['rays_d'],
                                            bg_color=sub_e['bg_color'],
                                            near_far=sub_e['near_fars'],
                                            timestamps=sub_e['timestamps'])
                fwd_out_rgb = torch.cat(
                    (fwd_out_rgb, sub_fwd_out['rgb'].detach().cpu()))
                del sub_e
            if (sub_e_idx != e['rays_o'].shape[0] - bs - 1):
                sub_e_idx += bs
                sub_e = {}
                sub_e['rays_o'] = e['rays_o'][sub_e_idx:].to(device)
                sub_e['rays_d'] = e['rays_d'][sub_e_idx:].to(device)
                sub_e['bg_color'] = e['bg_color'].repeat(sub_e['rays_o'].shape[0], 1).to(device)
                if (config['dataset_name'] == 'nerf_synthetic'):
                    sub_e['near_fars'] = e['near_fars'].repeat(sub_e['rays_o'].shape[0], 1).to(device)
                    sub_e['timestamps'] = None
                else:
                    sub_e['near_fars'] = e['near_fars'][sub_e_idx:].to(device)
                    sub_e['timestamps'] = e['timestamps'][sub_e_idx:].to(device)

                sub_fwd_out = kplanes_model(sub_e['rays_o'],
                                            sub_e['rays_d'],
                                            bg_color=sub_e['bg_color'],
                                            near_far=sub_e['near_fars'],
                                            timestamps=sub_e['timestamps'])
                fwd_out_rgb = torch.cat((fwd_out_rgb, sub_fwd_out['rgb'].detach().cpu()))
                del sub_e

            pred_img = fwd_out_rgb.clamp(0, 1).mul(255.0).byte()
            assert (pred_img.shape[0] == img_gt.shape[0] * img_gt.shape[1])
            pred_img = pred_img.view(e['img_h'], e['img_w'], 3)
            if eval_renders:
                tpred_img = pred_img / 255.0
                timg_gt = img_gt / 255.0
            else:
                mid = e['img_w'] // 2
                tpred_img = pred_img[:, :mid, :] / 255.0
                timg_gt = img_gt[:, :mid, :] / 255.0

            mses[f'img_{i}'] = torch.mean((tpred_img - timg_gt) ** 2).item()
            psnrs[f'img_{i}'] = psnr(tpred_img, timg_gt)
            ssims[f'img_{i}'] = ssim(tpred_img, timg_gt)
            lpipss[f'img_{i}'] = lpips_loss(tpred_img.permute(2, 0, 1).unsqueeze(0).to(device)*2-1, timg_gt.permute(2, 0, 1).unsqueeze(0).to(device)*2-1).item()

            total_psnr += psnrs[f'img_{i}']
            total_ssim += ssims[f'img_{i}']
            total_mse += mses[f'img_{i}']
            total_lpips += lpipss[f'img_{i}']

            # Saving
            if (epoch_nb is not None):
                save_image_from_tensor(
                    pred_img, filename=f"{config['output_dir']}/epoch_{epoch_nb}/img_{i}.png")
                save_image_from_tensor(
                    img_gt, filename=f"{config['output_dir']}/epoch_{epoch_nb}/img_{i}_gt.png")
            else:
                save_image_from_tensor(
                    pred_img, filename=f"{config['output_dir']}/img_{i}.png")
                save_image_from_tensor(
                    img_gt, filename=f"{config['output_dir']}/img_{i}_gt.png")

            pb.update(1)
        pb.close()

    len_psnrs = len(psnrs)
    len_ssims = len(ssims)
    len_mses = len(mses)
    len_lpipss = len(lpipss)
    psnrs['total'] = total_psnr
    psnrs['avg'] = total_psnr / len_psnrs
    ssims['total'] = total_ssim
    ssims['avg'] = total_ssim / len_ssims
    mses['total'] = total_mse
    mses['avg'] = total_mse / len_mses
    lpipss['total'] = total_lpips
    lpipss['avg'] = total_lpips / len_lpipss

    if (epoch_nb is not None):
        csv_filename = f"{config['output_dir']}/epoch_{epoch_nb}/psnrs.csv"
        csv_filename_ssim = f"{config['output_dir']}/epoch_{epoch_nb}/ssims.csv"
        csv_filename_mse = f"{config['output_dir']}/epoch_{epoch_nb}/mses.csv"
        csv_filename_lpips = f"{config['output_dir']}/epoch_{epoch_nb}/lpips.csv"
    else:
        csv_filename = f"{config['output_dir']}/psnrs.csv"
        csv_filename_ssim = f"{config['output_dir']}/ssims.csv"
        csv_filename_mse = f"{config['output_dir']}/mses.csv"
        csv_filename_lpips = f"{config['output_dir']}/lpips.csv"

    with open(csv_filename, 'w') as f:
        for key in psnrs.keys():
            f.write("%s,%s\n" % (key, psnrs[key]))

    with open(csv_filename_ssim, 'w') as f:
        for key in ssims.keys():
            f.write("%s,%s\n" % (key, ssims[key]))

    with open(csv_filename_mse, 'w') as f:
        for key in mses.keys():
            f.write("%s,%s\n" % (key, mses[key]))

    with open(csv_filename_lpips, 'w') as f:
        for key in lpipss.keys():
            f.write("%s,%s\n" % (key, lpipss[key]))
