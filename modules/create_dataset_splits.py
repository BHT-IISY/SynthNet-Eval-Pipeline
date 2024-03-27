""" Contains logic for creating dataset splits """

import os
import pandas as pd

from conf import paths
from conf._enums import Splits


def create_dataset_distribution_csv(dataset_csv: pd.DataFrame,
                                    save: bool = True,
                                    save_file: str = None) -> pd.DataFrame:
    # Save Dataset distribution
    train_distribution = dataset_csv.loc[dataset_csv['split'] == 'train'].groupby(
        'class_label', as_index=False)['image'].count().rename(columns={'image': 'n_train_images'})
    test_distribution = dataset_csv.loc[dataset_csv['split'] == 'test'].groupby(
        'class_label', as_index=False)['image'].count().rename(columns={'image': 'n_test_images'})
    dataset_distribution = train_distribution.join(test_distribution.set_index('class_label'), on='class_label')
    dataset_distribution['n_train_images'] = dataset_distribution['n_train_images'].fillna(0).astype('int64')
    dataset_distribution['n_test_images'] = dataset_distribution['n_test_images'].fillna(0).astype('int64')
    if save and save_file:
        dataset_distribution.to_csv(save_file, index=False)
    return dataset_distribution


def split_traintest_full():
    """ Use full datasets as train and test sets from config (no balancing or anything)
    """

    train_data = []
    for train_root_dir in paths.base_data_train_roots:
        train_ims_root = f'{train_root_dir}/images'

        for path, _, files in os.walk(train_ims_root):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    split_path = path.split('/')
                    im_path = f"{path}/{f}"
                    mesh = split_path[-1]
                    class_label = split_path[-2]
                    split = 'train'
                    train_data.append([im_path, class_label, mesh, split])

    # Get unique set of all train classes to ensure all test_classes are in our train set
    train_classes = set([im_data[1] for im_data in train_data])

    test_data = []
    for test_root_dir in paths.base_data_test_roots:
        test_ims_root = f'{test_root_dir}/images'

        for path, _, files in os.walk(test_ims_root):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    split_path = path.split('/')
                    im_path = f"{path}/{f}"
                    class_label = split_path[-1]
                    split = 'test'
                    if class_label in train_classes:
                        # NOTE: We don't have a mesh entity directory for test data.
                        test_data.append([im_path, class_label, "", split])

    traintest_full_dataset = pd.DataFrame(data=train_data + test_data,
                                          columns=['image', 'class_label', 'mesh', 'split'])

    for fe in paths.available_feature_extractors:
        ppaths = paths.get_paths_of_pair(fe, str(Splits.traintest_full))

        os.makedirs(ppaths.out_dir, exist_ok=True)
        traintest_full_dataset.to_csv(ppaths.dataset_csv, index=False)

        # Save Dataset distribution
        create_dataset_distribution_csv(dataset_csv=traintest_full_dataset,
                                        save=True,
                                        save_file=ppaths.dataset_distribution_csv)


def create_splits(seed: int = 42) -> None:
    if str(Splits.traintest_full) in paths.available_splits:
        split_traintest_full()
    # if str(Splits.syn2real_fitted) in paths.available_splits:
    #     split_syn2real_fitted()
    # if str(Splits.syn2syn) in paths.available_splits:
    #     split_syn2syn(seed=seed)


