import os, csv
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
import ast
import random
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm

import wilds
import configs.supported as supported
from utils import parse_bool, ParseKwargs
from configs.utils import populate_defaults

import subprocess
import shlex

split_dict = {'train': 0, 'val': 1, 'test': 2}


def train(config, dataset_version='all_data_with_identities.csv'):
    dataset = config.dataset
    algorithm = config.algorithm
    root_dir = config.root_dir
    frac = config.frac
    n_epochs = config.n_epochs
    log_dir = config.log_dir
    cmd = f'python examples/run_expt.py --dataset={dataset} --log_dir {log_dir} --algorithm={algorithm} ' + \
          f'--root_dir={root_dir} --frac={frac} --n_epochs={n_epochs} --dataset_version {dataset_version}'
    subprocess.run(shlex.split(cmd))


def evaluate(config, dataset_version):
    dataset = config.dataset
    algorithm = config.algorithm
    root_dir = config.root_dir
    frac = config.frac
    n_epochs = config.n_epochs
    log_dir = config.log_dir
    cmd = f'python examples/run_expt.py --dataset={dataset} --log_dir {log_dir} --algorithm={algorithm} ' + \
          f'--root_dir={root_dir} --frac={frac} --n_epochs={n_epochs} --dataset_version {dataset_version} --eval_only'
    subprocess.run(shlex.split(cmd))


def get_config():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    parser.add_argument('--split_scheme',
                        help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--dataset_version', type=str)
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to downloads the dataset if it does not exist in root_dir.')
    parser.add_argument('--version', default=None, type=str)
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--subsample_seed', type=int, default=1,
                        help='Subsampling the dataset randomly')

    # Algorithm
    parser.add_argument('--self_train_threshold', type=float, default=0.8)
    parser.add_argument('--self_train_rounds', type=int, default=3)

    # Optimization
    parser.add_argument('--n_epochs', type=int)

    # Evaluation
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)

    # Misc
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)

    config = parser.parse_args()
    return config


full_dataset_version_dict = {
    'amazon':'reviews.csv',
    'bdd100k':'',
    'camelyon17':'metadata.csv',
    'celebA':'list_attr_celeba.csv',
    'civilcomments':'all_data_with_identities.csv',
    'fmow':'rgb_metadata.csv',
    'iwildcam':'metadata.csv',
    'ogbmolpcba':'',
    'poverty':'dhs_metadata.csv',
    'py150':'',
    'sqf':'sqf.csv',
    'waterbirds':'metadata.csv',
    'yelp':'reviews.csv'
}


subsampled_dataset_version_dict = {
    'amazon':'subsample_amazon.csv',
    'bdd100k':'subsample_bdd100k.csv',
    'camelyon17':'subsample_camelyon17.csv',
    'celebA':'subsample_celebA.csv',
    'civilcomments':'subsample_civilcomments.csv',
    'fmow':'subsample_fmow.csv',
    'iwildcam':'subsample_iwildcam.csv',
    'ogbmolpcba':'subsample_ogbmolpcba.csv',
    'poverty':'subsample_poverty.csv',
    'py150':'subsample_py150.csv',
    'sqf':'subsample_sqf.csv',
    'waterbirds':'subsample_waterbirds.csv',
    'yelp':'subsample_yelp.csv'
}


labeled_dataset_version_dict = {
    'amazon':'labeled_amazon.csv',
    'bdd100k':'labeled_bdd100k.csv',
    'camelyon17':'labeled_camelyon17.csv',
    'celebA':'labeled_celebA.csv',
    'civilcomments':'labeled_civilcomments.csv',
    'fmow':'labeled_fmow.csv',
    'iwildcam':'labeled_iwildcam.csv',
    'ogbmolpcba':'labeled_ogbmolpcba.csv',
    'poverty':'labeled_poverty.csv',
    'py150':'labeled_py150.csv',
    'sqf':'labeled_sqf.csv',
    'waterbirds':'labeled_waterbirds.csv',
    'yelp':'labeled_yelp.csv'
}


label_key_dict = {
    'amazon':'y',
    'bdd100k':'',
    'camelyon17':'tumor',
    'celebA':'',
    'civilcomments':'toxicity',
    'fmow':'',
    'iwildcam':'y',
    'ogbmolpcba':'',
    'poverty':'wealthpooled',
    'py150':'input',
    'sqf':'found.weapon',
    'waterbirds':'y',
    'yelp':'y'
}


def subsample(metadata_df, config, TRAIN_SIZE, UNLABELED_TEST_SIZE, LABELED_TEST_SIZE, VAL_SIZE):
    subsampled_splits = []
    labeled_splits = []

    random.seed(config.subsample_seed)
    for split in split_dict:
        split_indices = metadata_df['split'] == split_dict[split]

        if split == 'train':
            indices = split_indices[split_indices == True]
            indices = list(indices.keys())
            if TRAIN_SIZE > 0:
                indices = random.sample(indices, TRAIN_SIZE)
            subsampled_splits.append(indices)
            labeled_splits.append(indices)
        elif split == 'test':
            indices = split_indices[split_indices == True]
            indices = list(indices.keys())
            if UNLABELED_TEST_SIZE > 0:
                unlabeled_indices = random.sample(indices, UNLABELED_TEST_SIZE)
            labeled_indices = [idx for idx in indices if idx not in unlabeled_indices]
            if LABELED_TEST_SIZE > 0:
                labeled_indices = random.sample(labeled_indices, LABELED_TEST_SIZE)
            subsampled_splits.append(unlabeled_indices)
            labeled_splits.append(labeled_indices)
        else:
            indices = split_indices[split_indices == True]
            indices = list(indices.keys())
            if VAL_SIZE > 0:
                indices = random.sample(indices, VAL_SIZE)
            subsampled_splits.append(indices)
            labeled_splits.append(indices)

    # here the test data will be the unlabeled split for which we want to collect
    # soft-max scores and pseudolabel
    all_indices = subsampled_splits[0] + subsampled_splits[1] + subsampled_splits[2]
    subsampled_df = metadata_df.loc[all_indices]
    subsampled_df.to_csv(f"{config.root_dir}/{config.dataset}_v1.0/subsample_{config.dataset}.csv",
                         index=False, header=list(subsampled_df.keys()))

    # here the test data will be the held out test set we want to evaluate the self
    # trained model on
    all_labeled_indices = labeled_splits[0] + labeled_splits[1] + labeled_splits[2]
    labeled_df = metadata_df.loc[all_labeled_indices]
    labeled_df.to_csv(f"{config.root_dir}/{config.dataset}_v1.0/labeled_{config.dataset}.csv",
                      index=False, header=list(labeled_df.keys()))


def main():
    config = get_config()
    dataset_version = subsampled_dataset_version_dict[config.dataset]
    labeled_dataset = labeled_dataset_version_dict[config.dataset]
    label_key = label_key_dict[config.dataset]
    if not dataset_version or not label_key:
        assert 0, print("Not implemented.")

    # SUBSAMPLE THE DATASET
    TRAIN_SIZE = 50000
    VAL_SIZE = -1
    UNLABELED_TEST_SIZE = 50000
    LABELED_TEST_SIZE = -1
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)
    metadata_df = full_dataset._metadata_df
    subsample(metadata_df, config, TRAIN_SIZE, UNLABELED_TEST_SIZE, LABELED_TEST_SIZE, VAL_SIZE)

    # self-train rounds
    for round in range(config.self_train_rounds):
        print(f"ROUND {round} OF SELF TRAINING WITH DATASET {dataset_version} \n\n")
        
        # train the model using current pseudolabeled splits
        train(config, dataset_version=dataset_version)

        # run eval on original splits with the model from the last round
        print(f"ROUND {round} EVALUATING MODEL\n\n")
        evaluate(config, dataset_version=labeled_dataset)

        # load the predictions (softmax scores)
        keys = ['idx', 'scores']
        probs_df = pd.read_csv(os.path.join(f"{config.log_dir}/{config.dataset}_split:test_seed:0_epoch:best_prob.csv"),
                               index_col=False, names=keys)

        # collect predictions on the resulting test data, if above the threshold
        print(f"DETERMINING CONFIDENT PREDICTIONS:")
        confident_indices = {}
        for ind, row in tqdm(probs_df.iterrows()):
            idx = row['idx']
            scores = ast.literal_eval(row['scores'])
            if max(scores) > config.self_train_threshold:
                pseudolabel = scores.index(max(scores))
                confident_indices[idx] = pseudolabel
        print(f"CONFIDENTLY LABELED EXAMPLES: {len(confident_indices)} OF {len(probs_df)}.\n\n")

        # update the dataset for the next round by loading current dataset_version and saving new version
        print(f"UPDATING SPLITS AND LABELS")
        full_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.version,
            root_dir=config.root_dir,
            download=config.download,
            split_scheme=config.split_scheme,
            dataset_version=dataset_version,
            **config.dataset_kwargs)
        metadata_df = full_dataset._metadata_df

        for ind, _ in confident_indices.items():
            assert ind in metadata_df.index.values.tolist(), print(metadata_df.index.values.tolist(), ind)

        pseudolabels = []
        splits = []
        ids = []
        num_train_examples = 0
        for ind, row in tqdm(metadata_df.iterrows()):
            ids.append(ind)
            if ind in confident_indices:
                splits.append(split_dict['train'])
                pseudolabels.append(confident_indices[ind])
                num_train_examples += 1
            else:
                splits.append(row['split'])
                pseudolabels.append(row[label_key])
                if row['split'] == split_dict['train']:
                    num_train_examples += 1
        assert len(metadata_df) == len(pseudolabels)
        assert len(metadata_df) == len(splits)
        
        print(f"TRAIN SET SIZE: {num_train_examples}\n\n")
        metadata_df['split'] = splits
        metadata_df[label_key] = pseudolabels
        metadata_df['id'] = ids
        dataset_version = f"iter{round+1}.csv"
        metadata_df.to_csv(f"{config.root_dir}/{config.dataset}_v1.0/{dataset_version}", index=False, header=list(metadata_df.keys()))

        # if len(metadata_df[metadata_df['split'] == split_dict['test']]) == 0:
        #     print("Pseudo labeled all examples! TODO, how to make it train without test examples.")
        #     break


if __name__=='__main__':
    main()