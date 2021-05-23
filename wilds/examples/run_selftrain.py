import os, csv
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
import ast
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm

import wilds
import configs.supported as supported
from utils import parse_bool, ParseKwargs
from configs.utils import populate_defaults

import subprocess
import shlex


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


def evaluate(config):
    dataset = config.dataset
    algorithm = config.algorithm
    root_dir = config.root_dir
    frac = config.frac
    n_epochs = config.n_epochs
    log_dir = config.log_dir
    cmd = f'python examples/run_expt.py --dataset={dataset} --log_dir {log_dir} --algorithm={algorithm} ' + \
          f'--root_dir={root_dir} --frac={frac} --n_epochs={n_epochs} --eval_only'
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

    # Algorithm
    parser.add_argument('--self_train_threshold', type=float, default=0.8)
    parser.add_argument('--self_train_rounds', type=int, default=2)

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


dataset_version_dict = {
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


def main():
    config = get_config()
    dataset_version = dataset_version_dict[config.dataset]
    if not dataset_version:
        assert 0, print("Not implemented.")

    # self-train rounds
    for round in range(config.self_train_rounds):
        print(f"ROUND {round} OF SELF TRAINING WITH DATASET {dataset_version} \n\n")
        
        # train the model using current pseudolabeled splits
        train(config, dataset_version=dataset_version)

        # run eval on original splits with the model from the last round
        print("EVALUATING LAST ITERATION MODEL\n\n")
        evaluate(config)

        # load the predictions (softmax scores)
        keys = ['idx', 'scores']
        probs_df = pd.read_csv(os.path.join(f"{config.log_dir}/civilcomments_split:test_seed:0_epoch:best_prob.csv"),
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
            assert ind in metadata_df.index.values.tolist()

        pseudolabels = []
        splits = []
        ids = []
        num_train_examples = 0
        for ind, row in tqdm(metadata_df.iterrows()):
            ids.append(ind)
            if ind in confident_indices:
                splits.append(0)
                pseudolabels.append(confident_indices[ind])
                num_train_examples += 1
            else:
                splits.append(row['split'])
                pseudolabels.append(row['toxicity'])
                if row['split'] == 0:
                    num_train_examples += 1
        assert len(metadata_df) == len(pseudolabels)
        assert len(metadata_df) == len(splits)
        
        print(f"TRAIN SET SIZE: {num_train_examples}\n\n")
        metadata_df['split'] = splits
        metadata_df['toxicity'] = pseudolabels
        metadata_df['id'] = ids
        dataset_version = f"iter{round+1}.csv"
        metadata_df.to_csv(f"{config.root_dir}/{config.dataset}_v1.0/{dataset_version}", index=False, header=list(metadata_df.keys()))

# Example Command:
# python examples/run_selftrain.py --dataset civilcomments --dataset_version iter0.csv --log_dir selftrain_test --root_dir data --split_scheme official --frac 0.002 --n_epochs 1 --algorithm ERM
if __name__=='__main__':
    main()