![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Contrastive Feature Loss for Image Prediction

We provide a PyTorch implementation of our contrastive feature loss presented in:

[Contrastive Feature Loss for Image Predcition](https://arxiv.org/abs/2111.06934)
[Alex Andonian](https://www.alexandonian.com), [Taesung Park](https://taesung.me/), [Bryan Russell](https://bryanrussell.org/), [Phillip Isola](http://web.mit.edu/phillipi/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Richard Zhang](https://richzhang.github.io/)

Presented in AIM Workshop at ICCV 2021

## Prerequisites

- Linux or macOS
- Python 3.6+
- CPU or NVIDIA GPU + CUDA CuDNN

**Table of Contents:**

1. [Setup](#setup)
2. [Dataset Preprocessing](#dataset-preparation)
3. [Training](#training-models)
4. [Evaluating and Visualizing](#evaluatating-and-visualizing)

### Setup

- Clone this repo:

```bash
git clone https://github.com/alexandonian/contrastive-feature-loss.git
cd contrastive-feature-loss
```

- Create python virtual environment

- Install a recent version of [PyTorch](https://pytorch.org/get-started/locally/) and other dependencies specified below.

We highly recommend that you install additional dependencies in an isolated python virtual environment (of your choosing). For Conda+pip users, you can create a new conda environment and then pip install dependencies with the following snippet:

```bash
ENV_NAME=contrastive-feature-loss
conda create --name $ENV_NAME python=3.8
conda activate $ENV_NAME
pip install -r requirements.txt
```

Alternatively, you can create a new Conda environment in one command using `conda env create -f environment.yml`, followed by `conda activate contrastive-feature-loss` to activate the environment.

This code also requires the Synchronized-BatchNorm-PyTorch rep.

```bash
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset Preparation

For COCO-Stuff, Cityscapes or ADE20K, the datasets must be downloaded beforehand. Please download them on the respective webpages. In the case of COCO-stuff, we put a few sample images in this code repo.

**Preparing COCO-Stuff Dataset**. The dataset can be downloaded [here](https://github.com/nightrome/cocostuff). In particular, you will need to download train2017.zip, val2017.zip, stuffthingmaps_trainval2017.zip, and annotations_trainval2017.zip. The images, labels, and instance maps should be arranged in the same directory structure as in `datasets/coco_stuff/`. In particular, we used an instance map that combines both the boundaries of "things instance map" and "stuff label map". To do this, we used a simple script `datasets/coco_generate_instance_map.py`. Please install `pycocotools` using `pip install pycocotools` and refer to the script to generate instance maps.

**Preparing ADE20K Dataset**. The dataset can be downloaded [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), which is from [MIT Scene Parsing BenchMark](http://sceneparsing.csail.mit.edu/). After unzipping the datgaset, put the jpg image files `ADEChallengeData2016/images/` and png label files `ADEChallengeData2016/annotatoins/` in the same directory.

There are different modes to load images by specifying `--preprocess_mode` along with `--load_size`. `--crop_size`. There are options such as `resize_and_crop`, which resizes the images into square images of side length `load_size` and randomly crops to `crop_size`. `scale_shortside_and_crop` scales the image to have a short side of length `load_size` and crops to `crop_size` x `crop_size` square. To see all modes, please use `python train.py --help` and take a look at `data/base_dataset.py`. By default at the training phase, the images are randomly flipped horizontally. To prevent this use `--no_flip`.

## Training Models

Models can be trained with the following steps.

1. Ensure you have prepared the dataset of interest using the instruction above. To train on the one of the datasets listed above, you can download the datasets and use `--dataset_mode` option, which will choose which subclass of `BaseDataset` is loaded. For custom datasets, the easiest way is to use `./data/custom_dataset.py` by specifying the option `--dataset_mode custom`, along with `--label_dir [path_to_labels] --image_dir [path_to_images]`. You also need to specify options such as `--label_nc` for the number of label classes in the dataset, `--contain_dontcare_label` to specify whether it has an unknown label, or `--no_instance` to denote the dataset doesn't have instance maps.

2. Run the training script with the following command:

```bash
# To train on the COCO dataset, for example.
python train.py --name [experiment_name] --dataset_mode coco --dataroot [path_to_coco_dataset]

# To train on your own custom dataset
python train.py --name [experiment_name] --dataset_mode custom --label_dir [path_to_labels] --image_dir [path_to_images] --label_nc [num_labels]
```

There are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.

The training logs are stored in json format in `[checkpoints_dir]/[experiment_name]/log.json`, with sample generations populated in `[checkpoint_dir]/[experiment_name]/web`.
We provide support for both Tensorboard (with `--tf_log`) and Weights & Biases (with `--use_wandb`) experiment tracking.

## Evaluating and Visualizing

In order to evaluate and visualize the generations of a trained model, run `test.py` in a similar manner, specifying the name of the experiment, the dataset and its path:

```bash
python test.py --name [name_of_experiment] --dataset_mode [dataset_mode] --dataroot [path_to_dataset]
```

where `[name_of_experiment]` is the directory name of the checkpoint created during training. If you are running on CPU mode, append `--gpu_ids -1`.

 <!-- where `[name_of_experiment]` is the directory name of the checkpoint created during training, which should be one of `coco_pretrained`, `ade20k_pretrained`, and `cityscapes_pretrained`. `[dataset]` can be one of `coco`, `ade20k`, and `cityscapes`, and `[path_to_dataset]`, is the path to the dataset. If you are running on CPU mode, append `--gpu_ids -1`. -->

Use `--results_dir` to specify the output directory. `--num_test` will specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

<!--
1. Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ/view?usp=sharing), save it in 'checkpoints/', and run

    ```bash
    cd checkpoints
    tar xvf checkpoints.tar.gz
    cd ../
    ``` -->

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `trainers/contrastive_pix2pix_trainer.py`: harnesses and reports the progress of training of contrastive model.
- `models/contrastive_pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `models/networks/loss`: contains proposed PatchNCE loss
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

## Options

This code repo contains many options. Some options belong to only one specific model, and some options have different default values depending on other options. To address this, the `BaseOption` class dynamically loads and sets options depending on what model, network, and datasets are used. This is done by calling the static method `modify_commandline_options` of various classes. It takes in the`parser` of `argparse` package and modifies the list of options. For example, since COCO-stuff dataset contains a special label "unknown", when COCO-stuff dataset is used, it sets `--contain_dontcare_label` automatically at `data/coco_dataset.py`. You can take a look at `def gather_options()` of `options/base_options.py`, or `models/network/__init__.py` to get a sense of how this works.

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2111.06934).

```bash
@inproceedings{andonian2021contrastive,
  title={Contrastive Feature Loss for Image Prediction},
  author={Andonian, Alex and Park, Taesung and Russell, Bryan and Isola, Phillip and Zhu, Jun-Yan and Zhang, Richard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1934--1943},
  year={2021}
}
```

## Acknowledgments

This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [SPADE](https://github.com/NVlabs/SPADE) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation). We thank Jiayuan Mao for his Synchronized Batch Normalization code.
