# CS329D_SelfTraining
Course Project


Example commands:

``python examples/run_selftrain.py --dataset civilcomments --log_dir selftrain_test_civil --root_dir data --split_scheme official --n_epochs 3 --algorithm ERM``
``python examples/run_selftrain.py --dataset amazon --log_dir selftrain_test_amazon --root_dir data --split_scheme official --algorithm ERM --data_dir baseline0.8thresh``

``python examples/run_expt.py --dataset civilcomments --algorithm ERM --root_dir data  --log_dir erm_civil --dataset_version labeled_civilcomments.csv``
``python examples/run_expt.py --dataset amazon --algorithm ERM --root_dir data  --log_dir erm_amazon --dataset_version baseline0.8thresh/labeled_amazon.csv``

For fast iteration (just using a subset of the data) you can add the flags:
``--frac 0.002 --n_epochs 1``
The following flags are for the self training threshold and number of iterations:
``--self_train_threshold 0.8 --self_train_rounds 2``
Overview: ``run_selftrain.py`` loads the original splits to train a model, then adds test examples based on their prediction confidence. Then saves the new splits to a new file and trains again on the new splits.