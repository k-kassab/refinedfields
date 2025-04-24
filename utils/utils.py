import os
import sys
import yaml
import numpy as np
import PIL
import torch
import random
import subprocess
import os

CONFIG_PATH = "./configs"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, filename=None):
    if filename is None : 
        filename = f"{config['log_dir']}/config.yaml"
    file=open(filename, "w")
    yaml.dump(config, file)
    file.close()

def save_image_from_tensor(t, filename="saved_image.jpg"):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    rgb_img = np.array(t, dtype=np.uint8)
    rgb_img = PIL.Image.fromarray(rgb_img)
    rgb_img = rgb_img.save(filename)

def init_dloader_random(_):
    seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    np.random.seed(seed)
    random.seed(seed)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def prepare_paths(config):
    config['output_dir'] = config['output_dir'] + config['expname']
    config['cache_dir'] = config['cache_dir'] + config['expname']
    config['log_dir'] = config['log_dir'] + config['expname']
    config['checkpoint_dir'] = config['checkpoint_dir'] + config['expname']

    if not os.path.isdir(config['output_dir']):
        os.makedirs(config['output_dir'])
    if not os.path.isdir(config['cache_dir']):
        os.makedirs(config['cache_dir'])
    if not os.path.isdir(config['log_dir']):
        os.makedirs(config['log_dir'])

    return config

def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
    return out

def get_free_gpu_indices():
    # code from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/12
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]

def modify_layer(source_layer:torch.nn.Conv2d, 
                 in_channels:int=None,
                 out_channels:int=None):
    if in_channels is None:
        in_channels = source_layer.in_channels
    if out_channels is None:
        out_channels = source_layer.out_channels

    new_layer = torch.nn.Conv2d(
        in_channels, 
        out_channels,
        kernel_size=source_layer.kernel_size, 
        stride=source_layer.stride, 
        padding=source_layer.padding,
        dtype=source_layer.weight.dtype,
        device=source_layer.weight.device
    )

    return new_layer