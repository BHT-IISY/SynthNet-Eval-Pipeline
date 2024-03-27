""" Contains functions to evaluate the results of a split-feature-extractor-pair """

import os
import faiss
import numpy as np
import pandas as pd

from conf import paths


def run_retrieval(dataset_df: pd.DataFrame, feature_map: pd.DataFrame, index_images: pd.DataFrame, search_index: faiss.Index) -> pd.DataFrame:
    """Prepare eval data DataFrame by  for the given parameters.

    Args:
        dataset_df (pd.DataFrame): DataFrame containing the image names, class_labels and whether they are in the test or train split
        feature_map (pd.DataFrame): DataFrame containing the precalculated features of the images contained in dataset_df
        index_images (pd.DataFrame): DataFrame containing the image names in the same order as in the search_index
        search_index (faiss.Index): Visual search index, used to evaluate the runs search performance

    Returns:
        retrieval_df (pd.DataFrame): DataFrame containing the results of the evaluation
    """
    ## PREPARE retrieval_df
    # Load img:feature map and convert to dataframe to merge features into retrieval_df
    features = feature_map.values.tolist()
    feature_map = feature_map.drop(columns=feature_map.columns)
    feature_map['feature_vector'] = features

    index_images['faiss_id'] = index_images.index
    index_images = index_images.merge(dataset_df[['image', 'class_label', 'mesh']], on="image", how="inner")

    retrieval_df = dataset_df.join(feature_map, on='image', how='inner')
    retrieval_df = retrieval_df.rename(columns={'class_label': 'y_true'})
    retrieval_df = retrieval_df[['image', 'split', 'feature_vector', 'y_true']]
    retrieval_df = retrieval_df.reset_index(drop=True)
    retrieval_df["y_pred_faiss_id"] = np.nan
    retrieval_df["y_pred_faiss_id"] = retrieval_df["y_pred_faiss_id"].astype(object)
    retrieval_df["y_pred_distance"] = np.nan
    retrieval_df["y_pred_distance"] = retrieval_df["y_pred_distance"].astype(object)
    retrieval_df["y_pred_class"] = np.nan
    retrieval_df["y_pred_class"] = retrieval_df["y_pred_class"].astype(object)
    retrieval_df["y_pred_im"] = np.nan
    retrieval_df["y_pred_im"] = retrieval_df["y_pred_im"].astype(object)
    retrieval_df["y_pred_mesh"] = np.nan
    retrieval_df["y_pred_mesh"] = retrieval_df["y_pred_mesh"].astype(object)

    ## MAKE QUERIES
    # get all classes in test set
    class_labels_train = dataset_df.loc[dataset_df["split"] == "train"]["class_label"].unique()
    class_labels_test = dataset_df.loc[dataset_df["split"] == "test"]["class_label"].unique()
    # get classes and item counts
    k_class_map = dataset_df.loc[(dataset_df["split"] == "train")]["class_label"].value_counts()
    for class_label in class_labels_test:
        # set number of neighbours (k) to retrieve for specific class
        k = int(k_class_map[class_label])
        # get embeddings for class
        # load as 2D array

        # collect images and their feature vectors
        q_data = retrieval_df.loc[retrieval_df["y_true"] == class_label]
        q_images = q_data['image'].values
        q_features = np.array([np.array(feat_vec) for feat_vec in q_data['feature_vector'].values]).astype(np.float32)
        # Perform Queries (all queries for current class)
        class_distances, class_neighbours = search_index.search(q_features, k)
        # Store retrievals for each query
        for i, (k_distances, k_neighbours) in enumerate(zip(class_distances, class_neighbours)):
            y_pred_faiss_id = k_neighbours
            y_pred_distance = k_distances
            y_pred_class = []
            y_pred_mesh = []
            y_pred_im = []
            for neighbour in k_neighbours:
                # loc returns a list of items, in this case it's always 1 item
                row_neighbour = index_images.loc[index_images['faiss_id'] == neighbour]
                y_pred_class += row_neighbour['class_label'].tolist()
                y_pred_mesh += row_neighbour['mesh'].tolist()
                y_pred_im += row_neighbour['image'].tolist()
            # y_pred_class = np.concatenate([
            #     index_images.loc[index_images['faiss_id'] == neighbour]['class_label'].values
            #     for neighbour in k_neighbours
            # ])
            # y_pred_mesh = np.concatenate(
            #     [index_images.loc[index_images['faiss_id'] == neighbour]['mesh'].values for neighbour in k_neighbours])
            # y_pred_im = np.concatenate(
            #     [index_images.loc[index_images['faiss_id'] == neighbour]['image'].values for neighbour in k_neighbours])

            # NOTE  Pandas is not designed to store array like objects in cells.
            #       Its should be avoided in general!
            # Get index of row for specific image
            df_i = retrieval_df.index[retrieval_df["image"] == q_images[i]].values[0]
            # Assign lists
            retrieval_df.at[df_i, "y_pred_faiss_id"] = y_pred_faiss_id
            retrieval_df.at[df_i, "y_pred_distance"] = y_pred_distance
            retrieval_df.at[df_i, "y_pred_class"] = y_pred_class
            retrieval_df.at[df_i, "y_pred_mesh"] = y_pred_mesh
            retrieval_df.at[df_i, "y_pred_im"] = y_pred_im

    return retrieval_df


def get_retrieval_results(split: str, feature_extractor: str) -> pd.DataFrame:
    """Evaluate the query results of a given split-feature-extractor-pair using faiss.

    Args:
        split (str): Name of the split
        feature_extractor (str): name of the feature extractor

    Returns: 
        config (dict): modified dictionary containing paths to the eval results
    """
    ppaths = paths.get_paths_of_pair(feature_extractor, split)

    os.makedirs(ppaths.retrieval_results_out_dir, exist_ok=True)

    # Load files
    dataset = pd.read_csv(ppaths.dataset_csv)
    feature_map = pd.DataFrame.from_dict(pd.read_pickle(ppaths.features_test_pickle), orient='index')
    index_images = pd.DataFrame(np.load(ppaths.index_images_npy), columns=['image'])
    search_index = faiss.read_index(ppaths.index_faiss)

    retrieval_df = run_retrieval(
        dataset_df=dataset,
        feature_map=feature_map,
        index_images=index_images,
        search_index=search_index,
    )

    retrieval_df.to_pickle(ppaths.img_retrieval_df_pickle)

    return retrieval_df
