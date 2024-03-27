"""Extract features and build a FAISS Search Index from those features"""

from typing import Tuple
import faiss
import numpy as np

from conf import paths
from modules.feature_extraction import create_feature_maps


def build_index(dataset_split: str,
                feature_extractor_name: str,
                feature_extractor_path: str,
                feature_extractor_dims: int,
                feature_map: dict,
                save: bool = True) -> Tuple[faiss.Index, np.ndarray]:
    """Build a FAISS IndexFlatL2 Search Index based on the features, the values of feature_map, with the dimensionality of feature_dims

    Args:
        feature_map (dict): Dictionary containing image names as keys and features as values
        feature_dims (int): Dimensionality of the features
        out_dir (str): Directory, where the resulting FAISS index and an array containing the mapping of filename to indices shall be saved
        save (bool, optional): True if results shall be saved to out_dir, False if not. Defaults to True.

    Returns:
        index, filenames (Tuple[faiss.Index, np.ndarray]): The FAISS IndexFlatL2, the image filenames
    """
    ppaths = paths.get_paths_of_pair(feature_extractor_name, dataset_split)

    feat_map = feature_map or create_feature_maps(
        feature_extractor_name=feature_extractor_name,
        feature_extractor_path=feature_extractor_path,
        dataset_split=dataset_split,
    )

    ims_fn, im_features = zip(*feat_map.items())
    ims_fn = np.array(ims_fn)
    np.save(ppaths.index_images_npy, ims_fn)

    index_flat = faiss.IndexFlatL2(feature_extractor_dims)
    im_features = np.array(im_features)
    index_flat.add(im_features)  # pylint: disable=no-value-for-parameter
    if save:
        faiss.write_index(index_flat, ppaths.index_faiss)
    return index_flat
