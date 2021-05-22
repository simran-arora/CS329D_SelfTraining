<p align='center'>
  <img width='40%' src='https://wilds.stanford.edu/WILDS_cropped.png' />
</p>

--------------------------------------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/wilds)](https://pypi.org/project/wilds/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/p-lambda/wilds/blob/master/LICENSE)

## Overview
WILDS is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping.

The WILDS package contains:
1. Data loaders that automatically handle data downloading, processing, and splitting, and
2. Dataset evaluators that standardize model evaluation for each dataset.

In addition, the example scripts contain default models, allowing new algorithms to be easily added and run on all of the WILDS datasets.

For more information, please read [our paper](https://arxiv.org/abs/2012.07421) or visit [our website](https://wilds.stanford.edu).
For questions and feedback, please post on the [discussion board](https://github.com/p-lambda/wilds/discussions).

## Installation

We recommend using pip to install WILDS:
```bash
pip install wilds
```

If you have already installed it, please check that you have the latest version:
```bash
python -c "import wilds; print(wilds.__version__)"
# This should print "1.1.0". If it doesn't, update by running:
pip install -U wilds
```

If you plan to edit or contribute to WILDS, you should install from source:
```bash
git clone git@github.com:p-lambda/wilds.git
cd wilds
pip install -e .
```

### Requirements
- numpy>=1.19.1
- ogb>=1.2.6
- outdated>=0.2.0
- pandas>=1.1.0
- pillow>=7.2.0
- pytz>=2020.4
- torch>=1.7.0
- torch-scatter>=2.0.5
- torch-geometric>=1.6.1
- tqdm>=4.53.0 

Running `pip install wilds` or `pip install -e .` will automatically check for and install all of these requirements
except for the `torch-scatter` and `torch-geometric` packages, which require a [quick manual install](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries).


### Default models
After installing the WILDS package, you can use the scripts in `examples/` to train default models on the WILDS datasets.
These scripts are not part of the installed WILDS package. To use them, you should clone the repo (assuming you did not install from source):
```bash
git clone git@github.com:p-lambda/wilds.git
```

To run these scripts, you will need to install these additional dependencies:

- torchvision>=0.8.1
- transformers>=3.5.0

All baseline experiments in the paper were run on Python 3.8.5 and CUDA 10.1.


## Using the example scripts

In the `examples/` folder, we provide a set of scripts that can be used to download WILDS datasets and train models on them.
These scripts are configured with the default models and hyperparameters that we used for all of the baselines described in our paper. All baseline results in the paper can be easily replicated with commands like:

```bash
python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data
python examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data
```

The scripts are set up to facilitate general-purpose algorithm development: new algorithms can be added to `examples/algorithms` and then run on all of the WILDS datasets using the default models.

The first time you run these scripts, you might need to download the datasets. You can do so with the `--download` argument, for example:
```
python examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --download
```

Alternatively, you can use the standalone `wilds/download_datasets.py` script to download the datasets, for example:

```bash
python wilds/download_datasets.py --root_dir data
```

This will download all datasets to the specified `data` folder. You can also use the `--datasets` argument to download particular datasets.

These are the sizes of each of our datasets, as well as their approximate time taken to train and evaluate the default model for a single ERM run using a NVIDIA V100 GPU.

| Dataset command | Modality | Download size (GB) | Size on disk (GB) | Train+eval time (Hours) |
|-----------------|----------|--------------------|-------------------|-------------------------|
| iwildcam        | Image    | 11                 | 25                | 7                       |
| camelyon17      | Image    | 10                 | 15                | 2                       |
| ogb-molpcba     | Graph    | 0.04               | 2                 | 15                      |
| civilcomments   | Text     | 0.1                | 0.3               | 4.5                     |
| fmow            | Image    | 50                 | 55                | 6                       |
| poverty         | Image    | 12                 | 14                | 5                       |
| amazon          | Text     | 6.6                | 7                 | 5                       |
| py150           | Text     | 0.1                | 0.8               | 9.5                     |

While the `camelyon17` dataset is small and fast to train on, we advise against using it as the only dataset to prototype methods on, as the test performance of models trained on this dataset tend to exhibit a large degree of variability over random seeds.

The image datasets (`iwildcam`, `camelyon17`, `fmow`, and `poverty`) tend to have high disk I/O usage. If training time is much slower for you than the approximate times listed above, consider checking if I/O is a bottleneck (e.g., by moving to a local disk if you are using a network drive, or by increasing the number of data loader workers). To speed up training, you could also disable evaluation at each epoch or for all splits by toggling `--evaluate_all_splits` and related arguments.

We have an [executable version](https://wilds.stanford.edu/codalab) of our paper on CodaLab that contains the exact commands, code, and data for the experiments reported in our paper, which rely on these scripts. Trained model weights for all datasets can also be found there.


## Using the WILDS package
### Data loading

The WILDS package provides a simple, standardized interface for all datasets in the benchmark.
This short Python snippet covers all of the steps of getting started with a WILDS dataset, including dataset download and initialization, accessing various splits, and preparing a user-customizable data loader.

```py
>>> from wilds import get_dataset
>>> from wilds.common.data_loaders import get_train_loader
>>> import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
>>> dataset = get_dataset(dataset='iwildcam', download=True)

# Get the training set
>>> train_data = dataset.get_subset('train',
...                                 transform=transforms.Compose([transforms.Resize((448,448)),
...                                                               transforms.ToTensor()]))

# Prepare the standard data loader
>>> train_loader = get_train_loader('standard', train_data, batch_size=16)

# Train loop
>>> for x, y_true, metadata in train_loader:
...   ...
```

The `metadata` contains information like the domain identity, e.g., which camera a photo was taken from, or which hospital the patient's data came from, etc.

### Domain information
To allow algorithms to leverage domain annotations as well as other
groupings over the available metadata, the WILDS package provides `Grouper` objects.
These `Grouper` objects extract group annotations from metadata, allowing users to
specify the grouping scheme in a flexible fashion.

```py
>>> from wilds.common.grouper import CombinatorialGrouper

# Initialize grouper, which extracts domain information
# In this example, we form domains based on location
>>> grouper = CombinatorialGrouper(dataset, ['location'])

# Train loop
>>> for x, y_true, metadata in train_loader:
...   z = grouper.metadata_to_group(metadata)
...   ...
```

The `Grouper` can be used to prepare a group-aware data loader that, for each minibatch, first samples a specified number of groups, then samples examples from those groups.
This allows our data loaders to accommodate a wide array of training algorithms,
some of which require specific data loading schemes.

```py
# Prepare a group data loader that samples from user-specified groups
>>> train_loader = get_train_loader('group', train_data,
...                                 grouper=grouper,
...                                 n_groups_per_batch=2,
...                                 batch_size=16)
```

### Evaluators

The WILDS package standardizes and automates evaluation for each dataset.
Invoking the `eval` method of each dataset yields all metrics reported in the paper and on the leaderboard.

```py
>>> from wilds.common.data_loaders import get_eval_loader

# Get the test set
>>> test_data = dataset.get_subset('test',
...                                 transform=transforms.Compose([transforms.Resize((224,224)),
...                                                               transforms.ToTensor()]))

# Prepare the data loader
>>> test_loader = get_eval_loader('standard', test_data, batch_size=16)

# Get predictions for the full test set
>>> for x, y_true, metadata in test_loader:
...   y_pred = model(x)
...   [accumulate y_true, y_pred, metadata]

# Evaluate
>>> dataset.eval(all_y_pred, all_y_true, all_metadata)
{'recall_macro_all': 0.66, ...}
```
Most `eval` methods take in predicted labels for `all_y_pred` by default, but the default inputs vary across datasets and are documented in the `eval` docstrings of the corresponding dataset class.

## Leaderboard
If you are developing new training algorithms and/or models on WILDS, please consider submitting them to our [public leaderboard](https://wilds.stanford.edu/leaderboard/).

## Citing WILDS
If you use WILDS datasets in your work, please cite [our paper](https://arxiv.org/abs/2012.07421) ([Bibtex](https://wilds.stanford.edu/assets/files/bibtex.md)):

- **WILDS: A Benchmark of in-the-Wild Distribution Shifts** (2021). Pang Wei Koh*, Shiori Sagawa*, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang.

Please also cite the original papers that introduce the datasets, as listed on the [datasets page](https://wilds.stanford.edu/datasets/).

## Acknowledgements
The design of the WILDS benchmark was inspired by the [Open Graph Benchmark](https://ogb.stanford.edu/), and we are grateful to the Open Graph Benchmark team for their advice and help in setting up WILDS.
