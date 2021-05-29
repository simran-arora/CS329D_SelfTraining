import os, csv
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
import ast
import math
import random
import json
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm

import wilds
import configs.supported as supported
from utils import parse_bool, ParseKwargs
from configs.utils import populate_defaults
import scipy
from scipy.stats import entropy

import subprocess
import shlex


split_dicts = {
    'civilcomments': {'train': 0, 'val': 1, 'test': 2},
    'amazon': {'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4}
}


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


# in the following, -1 indicates using the full split
split_sizes = {
    # amazon sizes in full: 245502 (train), 100050 (val), 46950 (id val), 100050 (test), 46950 (id test)
    'amazon': {'TRAIN': 50000, 'LABELED_TEST': -1, 'UNLABELED_TEST': 50000, 'VAL': 25000},
    # 'amazon': {'TRAIN': 50, 'LABELED_TEST': 50, 'UNLABELED_TEST': 50, 'VAL': 50},

    # civil comments sizes in full: ~200k, ~45k, ~130k
    'civilcomments': {'TRAIN': 50000, 'LABELED_TEST': -1, 'UNLABELED_TEST': 50000, 'VAL': -1},
}


datasetname_suffix = {
    'amazon':'v2.0',
    'civilcomments':'v1.0',
}


def get_config():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help="The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).")
    parser.add_argument('--data_dir', required=True,
                        help="The sub directory where to write out the subsample data for this experiment.")

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
    parser.add_argument('--confidence_condition', type=str, default='fixed_threshold')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--batch_size', type=int)

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


def train(config, dataset_version='all_data_with_identities.csv', log_dir=''):
    dataset = config.dataset
    algorithm = config.algorithm
    root_dir = config.root_dir
    batch_size = config.batch_size
    cmd = f'python examples/run_expt.py --dataset={dataset} --log_dir {log_dir} --algorithm={algorithm} ' + \
          f'--root_dir={root_dir} --dataset_version {dataset_version} --batch_size {batch_size}'
    subprocess.run(shlex.split(cmd))


def evaluate(config, dataset_version, log_dir=''):
    dataset = config.dataset
    algorithm = config.algorithm
    root_dir = config.root_dir
    batch_size = config.batch_size
    cmd = f'python examples/run_expt.py --dataset={dataset} --log_dir {log_dir} --algorithm={algorithm} ' + \
          f'--root_dir={root_dir} --dataset_version {dataset_version} --eval_only --batch_size {batch_size} ' \
          f'--save_best --save_last'
    subprocess.run(shlex.split(cmd))


def subsample(metadata_df, config, subsample_sizes, dataset_suffix):
    TRAIN_SIZE = subsample_sizes['TRAIN']
    UNLABELED_TEST_SIZE = subsample_sizes['UNLABELED_TEST']
    VAL_SIZE = subsample_sizes['VAL']
    LABELED_TEST_SIZE = subsample_sizes['LABELED_TEST']
    split_dict = split_dicts[config.dataset]
    subsampled_splits = []
    labeled_splits = []

    random.seed(config.subsample_seed)
    # train_splits = [split for split, num in split_dict.items() if 'train' in split]
    # test_splits = [split for split, num in split_dict.items() if 'test' in split]
    # val_splits = [split for split, num in split_dict.items() if 'val' in split]
    for split in split_dict:
        split_indices = metadata_df['split'] == split_dict[split]
        if split == 'train':
            indices = split_indices[split_indices == True]
            indices = list(indices.keys())
            if TRAIN_SIZE > 0 and len(indices) > TRAIN_SIZE:
                indices = random.sample(indices, TRAIN_SIZE)
            subsampled_splits.append(indices)
            labeled_splits.append(indices)
        elif split == 'test':
            indices = split_indices[split_indices == True]
            indices = list(indices.keys())
            unlabeled_indices = []
            if UNLABELED_TEST_SIZE > 0 and len(indices) > UNLABELED_TEST_SIZE:
                unlabeled_indices = random.sample(indices, UNLABELED_TEST_SIZE)

            labeled_indices = [idx for idx in indices if idx not in unlabeled_indices]
            if LABELED_TEST_SIZE > 0 and len(indices) > LABELED_TEST_SIZE:
                labeled_indices = random.sample(labeled_indices, LABELED_TEST_SIZE)
            subsampled_splits.append(unlabeled_indices)
            labeled_splits.append(labeled_indices)
        elif 'val' in split:
            indices = split_indices[split_indices == True]
            indices = list(indices.keys())
            if VAL_SIZE > 0 and len(indices) > VAL_SIZE:
                indices = random.sample(indices, VAL_SIZE)
            subsampled_splits.append(indices)
            labeled_splits.append(indices)

    # here the test data will be the unlabeled split for which we want to collect soft-max scores and pseudolabel
    all_indices = []
    for index_lst in subsampled_splits:
        all_indices += index_lst
    subsampled_df = metadata_df.loc[all_indices]
    subsampled_df['id'] = all_indices
    if not os.path.exists(f"{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}"):
        os.makedirs(f"{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}")
    subsampled_df.to_csv(f"{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/subsample_{config.dataset}.csv",
                         index=False, header=list(subsampled_df.keys()))

    # here the test data will be the held out test set we want to evaluate the self trained model on
    all_labeled_indices = []
    for index_lst in labeled_splits:
        all_labeled_indices += index_lst
    labeled_df = metadata_df.loc[all_labeled_indices]
    labeled_df['id'] = all_labeled_indices
    labeled_df.to_csv(f"{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/labeled_{config.dataset}.csv",
                      index=False, header=list(labeled_df.keys()))


# SCHEMES FOR SELECTING PSEUDOLABELS
def fixed_threshold(config, probs_df, metadata_df):
    confident_indices = {}
    for ind, row in tqdm(probs_df.iterrows()):
        idx = row['idx']
        scores = ast.literal_eval(row['scores'])
        if max(scores) > config.self_train_threshold:
            pseudolabel = scores.index(max(scores))
            confident_indices[idx] = pseudolabel
    return confident_indices


def fixed_group_proportion(config, probs_df, metadata_df):
    confident_indices = {}
    identity_vars = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']

    # get probs for each group
    group_probs = defaultdict(list)
    for ind, row in tqdm(probs_df.iterrows()):
        idx = row['idx']
        scores = ast.literal_eval(row['scores'])
        metadata_row = metadata_df[metadata_df.index == idx]
        if config.dataset == "civilcomments":
            for var in identity_vars:
                if metadata_row[var].values[0] != 0:
                    group_probs[var].append([idx, max(scores)])
        if config.dataset == 'amazon':
            group_probs[metadata_row['reviewerID'].values[0]].append([idx, max(scores)])

    # get max prob examples in each group
    all_selected_indices = []
    for group, prob_lst in group_probs.items():
        prob_lst = sorted(prob_lst, key=lambda x: x[1], reverse=True)
        num_to_take = math.floor(len(prob_lst)*config.self_train_threshold)
        prob_lst = prob_lst[:num_to_take]
        selected_indices = [tup[0] for tup in prob_lst if tup[0] not in all_selected_indices]
        group_probs[group] = selected_indices
        all_selected_indices.extend(selected_indices)

    # get the predictions for the selected examples and return
    for ind, row in tqdm(probs_df.iterrows()):
        idx = row['idx']
        metadata_row = metadata_df[metadata_df.index == idx]
        if config.dataset == "civilcomments":
            for var in identity_vars:
                if metadata_row[var].values[0] != 0:
                    group = var
                    break
        if config.dataset == 'amazon':
            group = metadata_row['reviewerID'].values[0]
        if idx in group_probs[group]:
            scores = ast.literal_eval(row['scores'])
            pseudolabel = scores.index(max(scores))
            confident_indices[idx] = pseudolabel
    return confident_indices


def entropy_ranked(config, probs_df, metadata_df):
    confident_indices = {}
    
    example_entropies = []
    for ind, row in tqdm(probs_df.iterrows()):
        idx = row['idx']
        scores = ast.literal_eval(row['scores'])
        prob = compute_entropy(scores)
        example_entropies.append([idx, prob])

    example_entropies = sorted(example_entropies, key=lambda x: x[1], reverse=False)
    num_to_take = math.floor(len(example_entropies)*config.self_train_threshold)
    example_entropies = example_entropies[:num_to_take]
    selected_indices = [tup[0] for tup in example_entropies]

    # get the predictions for the selected examples and return
    for ind, row in tqdm(probs_df.iterrows()):
        idx = row['idx']
        if idx in selected_indices:
            scores = ast.literal_eval(row['scores'])
            pseudolabel = scores.index(max(scores))
            confident_indices[idx] = pseudolabel
    return confident_indices


def compute_entropy(probs):
    confidence = entropy(list(probs), base=2)
    return confidence


def main():
    config = get_config()

    # make the location to save the run dataset (subsampled) and config
    dataset_suffix = datasetname_suffix[config.dataset]
    if not os.path.exists(f'{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/'):
        os.makedirs(f'{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/')
    with open(f'{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/config.json', 'w') as f:
        json.dump(vars(config), f)

    # obtain dataset names to use
    dataset_version = subsampled_dataset_version_dict[config.dataset]
    labeled_dataset = labeled_dataset_version_dict[config.dataset]
    label_key = label_key_dict[config.dataset]
    split_dict = split_dicts[config.dataset]
    if not dataset_version or not label_key:
        assert 0, print("Not implemented.")

    # SUBSAMPLE THE DATASET
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)
    metadata_df = full_dataset._metadata_df
    subsample_sizes = split_sizes[config.dataset]
    if not os.path.isfile(f"{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/{dataset_version}"):
        subsample(metadata_df, config, subsample_sizes, dataset_suffix)

    # self-train rounds
    for round in range(config.self_train_rounds):
        print(f"ROUND {round} OF SELF TRAINING WITH DATASET {config.data_dir}/{dataset_version} \n\n")

        # train the model using current pseudolabeled splits
        log_dir = f"{config.log_dir}/round{round}/{config.data_dir}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(config, dataset_version=f"{config.data_dir}/{dataset_version}", log_dir=f"{log_dir}")

        # run eval on original splits with the model from the last round
        print(f"ROUND {round} EVALUATING MODEL\n\n")
        evaluate(config, dataset_version=f"{config.data_dir}/{labeled_dataset}", log_dir=f"{log_dir}")

        # load the predictions (softmax scores)
        keys = ['idx', 'scores']
        probs_df = pd.read_csv(os.path.join(f"{log_dir}/{config.dataset}_split:test_seed:0_epoch:best_prob.csv"),
                               index_col=False, names=keys)

        # loading current dataset_version
        full_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.version,
            root_dir=config.root_dir,
            download=config.download,
            split_scheme=config.split_scheme,
            dataset_version=f"{config.data_dir}/{dataset_version}",
            **config.dataset_kwargs)
        metadata_df = full_dataset._metadata_df

        # collect predictions on the resulting test data, if above the threshold
        print(f"DETERMINING CONFIDENT PREDICTIONS:")
        if config.confidence_condition == 'fixed_threshold':
            confident_indices = fixed_threshold(config, probs_df, metadata_df)
        elif config.confidence_condition == 'fixed_group_proportion':
            confident_indices = fixed_group_proportion(config, probs_df, metadata_df)
        else:
            assert 0, print("INVALID CONFIDENCE CONDITION")
        print(f"CONFIDENTLY LABELED EXAMPLES: {len(confident_indices)} OF {len(probs_df)}.\n\n")

        # update the dataset for the next round
        print(f"UPDATING SPLITS AND LABELS")


        for ind, _ in confident_indices.items():
            assert ind in metadata_df.index.values.tolist(), \
                print(metadata_df.index.values.tolist(), ind)

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
        metadata_df.to_csv(f"{config.root_dir}/{config.dataset}_{dataset_suffix}/{config.data_dir}/{dataset_version}",
                           index=False, header=list(metadata_df.keys()))


if __name__=='__main__':
    main()