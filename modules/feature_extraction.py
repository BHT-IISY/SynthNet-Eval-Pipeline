""" Provides funcionality to extract features from images """

from typing import List
import os
import pickle
import pandas as pd
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel, ConvNextFeatureExtractor, ConvNextForImageClassification
from tqdm import tqdm

from conf import paths
from utils.im_utils import expand2square


def create_feature_maps(
    feature_extractor_name: str,
    feature_extractor_path: str,
    dataset_split: str,
) -> None:
    """Extracts the features using feature_extractor from the images corresponding to 
    the combination of run_name and dataset_split given. Then build a FAISS Search Index 
    based on these features.

    Args:
        feature_extractor (str): Feature extractor to be used in the extraction
        dataset_split (str): Name of the split, whose index shall be build
    """
    ppaths = paths.get_paths_of_pair(feature_extractor_name, dataset_split)
    os.makedirs(ppaths.out_dir, exist_ok=True)

    dataset_csv = pd.read_csv(ppaths.dataset_csv)

    # Extract features for train/test images
    feature_maps = {}
    for split in ["train", "test"]:
        data = dataset_csv.loc[dataset_csv['split'] == split]
        ims = data['image'].to_list()
        feature_map = extract_features(
            image_paths=ims,
            feature_extractor_name=feature_extractor_name,
            feature_extractor_path=feature_extractor_path,
            out_file=ppaths.get_features_pickle(split),
            save=True,
        )
        feature_maps[split] = feature_map

    return feature_maps['train']


def extract_features(
    image_paths: List[str],
    feature_extractor_name: str,
    feature_extractor_path: str,
    out_file: str,
    save: bool,
) -> dict:
    """Extract the features of images using feature_extractor from the given set of images in the directory image_dir with the filenames image_fns

    Args:
        image_paths (List[str]): Paths of the images
        feature_extractor (str): Name of the feature extractor to be used
        out_file (str): Filename ending in '.pickle', where the resulting features shall be saved
        save (bool): True if features shall be saved in 'out_file', False if not

    Returns:
        features (dict): Dictionary containing the filenames as keys and features as their corresponding values
    """
    if feature_extractor_name == 'vit_in21k':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        im_preprocessing = ViTFeatureExtractor(
            do_resize=True,
            size=224,
            # resample=Image.BILINEAR,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        ).from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model = model.to(device)
        features = {}
        for im_path in tqdm(image_paths):
            im = Image.open(f'{im_path}').convert('RGB')
            # NOTE Padding image (black) to square before resizing
            im = expand2square(im, (0, 0, 0))
            inputs = im_preprocessing(im, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
                outputs = outputs.to('cpu')
                # Transform to squeezed FLOAT32 numpy array
                outputs = outputs.numpy().astype('float32').squeeze()
                features[im_path] = outputs
        if save:
            with open(out_file, 'wb') as f:
                pickle.dump(features, f)
        return features
    # if feature_extractor_name == 'convnext_s':
    #     break

    elif feature_extractor_name == 'vit_in21k_finetuned':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        im_preprocessing = ViTFeatureExtractor(
            do_resize=True,
            size=224,
            # resample=Image.BILINEAR,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        ).from_pretrained(feature_extractor_path)
        model = ViTModel.from_pretrained(feature_extractor_path)
        model = model.to(device)
        features = {}
        for im_path in tqdm(image_paths):
            im = Image.open(f'{im_path}').convert('RGB')
            # NOTE Padding image (black) to square before resizing
            im = expand2square(im, (0, 0, 0))
            inputs = im_preprocessing(im, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
                outputs = outputs.to('cpu')
                # Transform to squeezed FLOAT32 numpy array
                outputs = outputs.numpy().astype('float32').squeeze()
                features[im_path] = outputs
        if save:
            with open(out_file, 'wb') as f:
                pickle.dump(features, f)
        return features
    elif feature_extractor_name == 'dino_vitb16_in1k':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        im_preprocessing = ViTFeatureExtractor(
            do_resize=True,
            size=224,
            # resample=Image.BILINEAR,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        ).from_pretrained('facebook/dino-vitb16')
        model = ViTModel.from_pretrained('facebook/dino-vitb16')
        model = model.to(device)
        features = {}
        for im_path in tqdm(image_paths):
            im = Image.open(f'{im_path}').convert('RGB')
            # NOTE Padding image (black) to square before resizing
            im = expand2square(im, (0, 0, 0))
            inputs = im_preprocessing(im, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
                outputs = outputs.to('cpu')
                # last_hidden_states = outputs.last_hidden_state
                # Transform to squeezed FLOAT32 numpy array
                outputs = outputs.numpy().astype('float32').squeeze()
                features[im_path] = outputs
        if save:
            with open(out_file, 'wb') as f:
                pickle.dump(features, f)
        return features