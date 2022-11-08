# CS329D Course Project: Exploring Self-training and Pre-training for Unsupervised Domain Adaptation

## Project Report.

The project report can be found [here](https://github.com/simran-arora/CS329D_SelfTraining/blob/main/CS329D_2021_Final_Project.pdf).


## Example commands:

### SELF TRAIN CMDs
``python examples/run_selftrain.py --dataset civilcomments --log_dir selftrain_test_civil --root_dir data --split_scheme official --algorithm ERM --data_dir baseline0.9thresh --batch_size 16 --self_train_threshold 0.9``

``python examples/run_selftrain.py --dataset amazon --log_dir selftrain_test_amazon --root_dir data --split_scheme official --algorithm ERM --data_dir baseline0.8thresh --batch_size 16 --self_train_threshold 0.8``

``python examples/run_selftrain.py --dataset amazon --log_dir selftrain_test_amazon --root_dir data --split_scheme official --algorithm ERM --data_dir group_prop_0.33 --batch_size 16 --self_train_threshold 0.33 --confidence_condition fixed_group_proportion``

### ERM CMDs
``python examples/run_expt.py --dataset civilcomments --algorithm ERM --root_dir data  --log_dir erm_civil --dataset_version labeled_civilcomments.csv``

``python examples/run_expt.py --dataset amazon --algorithm ERM --root_dir data  --log_dir erm_amazon --dataset_version baseline0.8thresh/labeled_amazon.csv --batch_size 16``

For fast iteration (just using a subset of the data) you can add the flags:
``--frac 0.002 --n_epochs 1``
The following flags are for the self training threshold and number of iterations:
``--self_train_threshold 0.8 --self_train_rounds 2``

The flat ``--confidence_condition`` specifies how to select confident pseudolabels. Implemented options are "fixed_group_proportion" and "fixed_threshold" (default).

Overview: ``run_selftrain.py`` loads the original splits to train a model, then adds test examples based on their prediction confidence. Then saves the new splits to a new file and trains again on the new splits.
