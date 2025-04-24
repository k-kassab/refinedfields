""" This script is used to launch the training of RefinedFields or its testing.

E.g. for training:  python main.py --config trevi.yaml --train_only --device 'cuda:0'
E.g. for testing:  python main.py --config trevi.yaml --test_only --device 'cuda:0'
"""

import argparse
import os
from typing import Dict, Any
import pprint
import torch

from copy import copy
from models.model_io import init_models, load_models
from utils.utils import load_config, init_dloader_random, query_yes_no, prepare_paths, save_config, get_free_gpu_indices
from test import test
from torch.utils.tensorboard import SummaryWriter

synthetic_scenes = [
    'lego',
    'chair',
    'drums',
    'ficus',
    'hotdog',
    'materials',
    'mic',
    'ship']
phototourism_scenes = [
    'sacre', 
    'trevi', 
    'brandenburg']


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        '--train_only',
        required=False,
        default=False,
        action='store_true')
    parser.add_argument(
        '--test_only',
        required=False,
        default=False,
        action='store_true')
    parser.add_argument(
        '--device',
        required=False,
        default=None,
        help="""Device to use. Can be int or string, e.g.: 'cpu', 'cuda:0', 0 , etc.
                        If not specified, will try to use a free gpu (if available)""")
    args = parser.parse_args()

    # Importing config
    print("\n********************************************************************************")
    print(f"Selected config file: {args.config}")
    print("--------------------------------------------------------------------------------")
    config: Dict[str, Any] = load_config(args.config)
    print("Selected configuration:")
    config = prepare_paths(config)
    pprint.pprint(config)
    print("********************************************************************************\n")

    # Setting up tensorboard
    writer = SummaryWriter(config['log_dir'])

    # Saving config
    if (not args.test_only):
        save_config(config)

    # Setting device
    if args.device is None:
        available_gpu = get_free_gpu_indices()
        if len(available_gpu) > 0:
            device_name = f"cuda:{available_gpu[0]}"
            print(f"Automatically selected device: {device_name}")
        else:
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Automatic device selection failed. Current device: {device_name}")
    else:
        device_name = args.device
    device = torch.device(device_name)

    # ------------- Data handling -------------
    # Preparing dataset
    if (config['scene_name'] in phototourism_scenes):
        from dataset.phototourism_dataset import PhotoTourismDataset
        tr_dset = PhotoTourismDataset(
            config['data_dir'],
            split='train',
            batch_size=config['batch_size'],
            scene_bbox=config['scene_bbox'],
            global_scale=config['global_scale'],
            global_translation=config['global_translation'],
            contraction=config['contract'],
            ndc=config['ndc'])
        ts_dset = PhotoTourismDataset(
            config['data_dir'],
            split='test',
            batch_size=config['batch_size'],
            scene_bbox=config['scene_bbox'],
            global_scale=config['global_scale'],
            global_translation=config['global_translation'],
            contraction=config['contract'],
            ndc=config['ndc'])
    elif (config['scene_name'] in synthetic_scenes):
        from dataset.synthetic_nerf_dataset import SyntheticNerfDataset
        tr_dset = SyntheticNerfDataset(config['data_dir'],
                                       split='train',
                                       downsample=config['data_downsample'],
                                       max_frames=None,
                                       batch_size=config['batch_size'])
        tr_dset.reset_iter()
        ts_dset = SyntheticNerfDataset(config['data_dir'],
                                       split='test',
                                       downsample=config['data_downsample'],
                                       max_frames=None,
                                       batch_size=config['batch_size'])
    else:
        print(f"Training for {config['scene_name']} not implemented.\n")
        return

    # Preparing dataloaders
    tr_loader = torch.utils.data.DataLoader(tr_dset,
                                            batch_size=None,
                                            num_workers=4,
                                            prefetch_factor=4,
                                            pin_memory=True,
                                            worker_init_fn=init_dloader_random,
                                            shuffle=True)
    # -----------------------------------------

    # Preparing some variables for the LowrankModel
    grid_config = [{'grid_dimensions': 2,
                    'input_coordinate_dim': config['nb_kplanes'],
                    'output_coordinate_dim': config['kplanes_channel_dim'],
                    'resolution': [config['kplanes_resolution'],
                                   config['kplanes_resolution'],
                                   config['kplanes_resolution']]}]
    extra_args = copy(config)
    extra_args.pop('global_scale', None)
    extra_args.pop('global_translation', None)
    config['grid_config'] = grid_config

    # ------------- Initializing models -------------
    optimizer_state_dict = None

    # Checking checkpoints
    l_epoch = 0
    if (config['load_checkpoint']):
        if not os.path.isdir(config['checkpoint_dir']):
            warn = f"WARNING: No checkpoint dir found: {config['checkpoint_dir']}, do you want to create it?"
            create_it = query_yes_no(warn, default="yes")
            if (create_it):
                os.makedirs(config['checkpoint_dir'])
                models = init_models(
                    device, config, grid_config, tr_dset, extra_args)
                k_planes_diffuser, kplanes_model, noisy_latent_code = models
            else:
                return
        else:
            models = init_models(
                device,
                config,
                grid_config,
                tr_dset,
                extra_args,
                create_attn_procs=False,
                ts_dset=ts_dset)
            k_planes_diffuser, kplanes_model, noisy_latent_code = models
            l_epoch, noisy_latent_code, k_planes_diffuser, kplanes_model, optimizer_state_dict, _, _ = load_models(
                config, k_planes_diffuser, kplanes_model, checkpoint=config['checkpoint_v'])
    else:
        if not os.path.isdir(config['checkpoint_dir']):
            os.makedirs(config['checkpoint_dir'])
            models = init_models(
                device,
                config,
                grid_config,
                tr_dset,
                extra_args)
            k_planes_diffuser, kplanes_model, noisy_latent_code = models
        else:
            nb_checkpoints = len(
                [
                    entry for entry in os.listdir(
                        config['checkpoint_dir']) if os.path.isfile(
                        os.path.join(
                            config['checkpoint_dir'],
                            entry))])
            if (nb_checkpoints > 0):
                warn = "WARNING: A checkpoint already exists, do you want to load it?"
                load_existing = query_yes_no(warn, default="yes")
                if (load_existing):
                    models = init_models(
                        device,
                        config,
                        grid_config,
                        tr_dset,
                        extra_args,
                        create_attn_procs=False,
                        ts_dset=ts_dset)
                    k_planes_diffuser, kplanes_model, noisy_latent_code = models
                    l_epoch, noisy_latent_code, k_planes_diffuser, kplanes_model, optimizer_state_dict, _, _ = load_models(
                        config, k_planes_diffuser, kplanes_model, checkpoint=config['checkpoint_v'])
            else:
                models = init_models(
                    device, config, grid_config, tr_dset, extra_args)
                k_planes_diffuser, kplanes_model, noisy_latent_code = models

    # -----------------------------------------------

    from train_alternate import train_alternate
    if (not (args.train_only or args.test_only)):
        train_alternate(config=config,
                        device=device,
                        l_epoch=l_epoch,
                        tr_loader=tr_loader,
                        noisy_latent_code=noisy_latent_code,
                        k_planes_diffuser=k_planes_diffuser,
                        kplanes_model=kplanes_model,
                        writer=writer,
                        ts_dset=ts_dset,
                        optimizer_state_dicts=optimizer_state_dict)
        test(config=config,
             device=device,
             k_planes_diffuser=k_planes_diffuser,
             kplanes_model=kplanes_model,
             ts_dset=ts_dset,
             epoch_nb=None)
    elif (args.train_only):
        train_alternate(config=config,
                        device=device,
                        l_epoch=l_epoch,
                        tr_loader=tr_loader,
                        noisy_latent_code=noisy_latent_code,
                        k_planes_diffuser=k_planes_diffuser,
                        kplanes_model=kplanes_model,
                        writer=writer,
                        ts_dset=ts_dset,
                        optimizer_state_dicts=optimizer_state_dict)
    else:
        test(config=config,
             device=device,
             kplanes_model=kplanes_model,
             ts_dset=ts_dset,
             epoch_nb=config['checkpoint_v'])


if __name__ == "__main__":
    main()
