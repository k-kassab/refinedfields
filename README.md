# RefinedFields: Radiance Fields Refinement for Planar Scene Representations

> Karim Kassab, Antoine Schnepf, Jean-Yves Franceschi, Laurent Caraffa, Jeremie Mary, Valérie Gouet-Brunet<br>
| [Project Page](https://refinedfields.github.io) |

![Figure](./assets/figure.svg)

## Setup
In this section we detail how to prepare the dataset and the environment for running RefinedFields. 

### Environment
Our code has been tested on:
- Python 3.7.16
- PyTorch 1.13.1
- CUDA 11.6
- An `NVIDIA Tesla A100` GPU

We recommend using Anaconda to create the environment using the provided `environment.yaml` file. Additionnaly, you will need to install TinyCudaNN by following the installation steps provided in [their repository](https://github.com/NVlabs/tiny-cuda-nn).

### Datasets
Our current implementation supports two datasets: `Photourism` and `nerf_synthetic`

#### Synthetic dataset
We utilize the same `nerf_synhetic` dataset from the original NeRF paper. Please refer to the [NeRF repo](https://github.com/bmild/nerf) to download it.
#### In-the-wild dataset
We utilize the same dataset as in [NeRF-W](https://nerf-w.github.io). Please refer to their repo to download it.

## Running
### Configuration
The configuration files for synthetic scenes and Phototourism scenes can be found in `configs/`.
Please update the data directory in the configuration file.

### Training
To start the training, run the following command: 
```
python main.py --config config.yaml --train_only
```
where config.yaml is the name of your configuration file.

### Testing
To test the model, run the following command 
```
python main.py --config config.yaml --test_only
```
where config.yaml is the name of your configuration file.

## License
This code is open-source. It is shared under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Some parts of the code modify the K-Planes codebase, which has been shared under a different license. The concerned files start with a special notice about this.

## Citation
```
@article{kassab2023refinedfields,
    title={RefinedFields: Radiance Fields Refinement for Unconstrained Scenes},
    author={Kassab, Karim and Schnepf, Antoine and Franceschi, Jean-Yves and Caraffa, Laurent and Mary, Jeremie and Gouet-Brunet, Val{\'e}rie},
    journal={arXiv preprint arXiv:2312.00639},
    year={2023}
}
   
```




