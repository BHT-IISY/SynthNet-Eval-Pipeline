""" Module to store file and directory paths for global usage"""
import os

__all__ = ["paths"]


def concat_paths(*args) -> str:
    return os.path.join(*args)


class PairPaths:
    """TODO docs"""

    # Default members
    fe_dir = None
    out_dir = None
    dataset_csv = None
    dataset_distribution_csv = None
    index_images_npy = None
    index_faiss = None
    features_test_pickle = None
    features_train_pickle = None
    retrieval_results_out_dir = None
    img_retrieval_df_pickle = None
    classification_report_csv = None

    def __init__(self, fe_out_dir: str, split_name: str) -> None:
        self.fe_dir = fe_out_dir
        self.out_dir = concat_paths(fe_out_dir, split_name)
        self.dataset_csv = concat_paths(self.out_dir, "dataset.csv")
        self.dataset_distribution_csv = concat_paths(self.out_dir, "dataset_distribution.csv")
        self.index_images_npy = concat_paths(self.out_dir, "index_images.npy")
        self.index_faiss = concat_paths(self.out_dir, "index.faiss")
        self.features_test_pickle = concat_paths(self.out_dir, "features_test.pickle")
        self.features_train_pickle = concat_paths(self.out_dir, "features_train.pickle")
        self.retrieval_results_out_dir = concat_paths(self.out_dir, "retrieval_results")
        self.img_retrieval_df_pickle = concat_paths(self.retrieval_results_out_dir, "img_retrieval_df.pickle")
        self.classification_report_csv = concat_paths(self.retrieval_results_out_dir, "classification_report.csv")

    def get_features_pickle(self, split: str) -> str:
        assert split in ["train", "test"]
        return getattr(self, f'features_{split}_pickle')


class Paths:
    """TODO docs"""

    # Default members
    base_out_dir = None
    base_run_dir = None
    base_run_config_path = None
    base_out_run_config_path = None
    base_data_train_roots = None
    base_data_test_root = None
    available_splits = []
    available_feature_extractors = []
    is_initialized = False

    def init_base_paths(
        self,
        out_dir: str,
        run_name: str,
        run_config_path: str,
        data_train_roots: str,
        data_test_roots: str,
    ) -> None:
        self.base_out_dir = out_dir
        self.base_run_dir = concat_paths(out_dir, run_name)
        self.base_run_config_path = run_config_path
        self.base_out_run_config_path = concat_paths(self.base_run_dir, "run_config.json")
        self.base_data_train_roots = data_train_roots
        self.base_data_test_roots = data_test_roots
        self.is_initialized = True

    def add_fe_split_pair(self, fe: str, split: str) -> None:
        assert self.is_initialized, "Need to initialize object before adding a pair"
        fe_base = f'fe_{fe}'
        fe_out_dir = concat_paths(self.base_run_dir, fe)
        combined_base = f'pair_{fe}_{split}'

        if hasattr(self, combined_base):
            return

        if not fe in self.available_feature_extractors:
            self.available_feature_extractors.append(fe)
            setattr(self, f'{fe_base}_out_dir', fe_out_dir)

        if not split in self.available_splits:
            self.available_splits.append(split)

        pair_paths = PairPaths(fe_out_dir=fe_out_dir, split_name=split)

        setattr(self, combined_base, pair_paths)

    def _get_pair(self, fe: str, split: str) -> PairPaths:
        return getattr(self, f'pair_{fe}_{split}')

    def get_paths_of_pair(self, fe: str, split: str) -> PairPaths:
        assert self.is_initialized, "Need to initialize object before retrieving a pair"
        if not fe in self.available_feature_extractors or not split in self.available_splits:
            return
        return self._get_pair(fe, split)

    def get_random_pair_by_split(self, split: str) -> PairPaths:
        assert self.is_initialized, "Need to initialize object before retrieving a pair"
        fe = self.available_feature_extractors[0]
        if not split in self.available_splits:
            return
        return self._get_pair(fe, split)


paths = Paths()