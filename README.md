# CS329D_SelfTraining
Course Project


Example command:

``python examples/run_selftrain.py --dataset civilcomments --log_dir selftrain_test_{fill in dataset} --root_dir data --split_scheme official --n_epochs 3 --algorithm ERM
``

For fast iteration (just using a subset of the data) you can add the flags:

``--frac 0.002 --n_epochs 1``

The following flags are for the self training threshold and number of iterations:

``--self_train_threshold 0.8 --self_train_rounds 2``

Overview: ``run_selftrain.py`` loads the original splits to train a model, then adds test examples based on their prediction confidence. Then saves the new splits to a new file and trains again on the new splits.