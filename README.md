# SynthNet Evaluation Pipeline

The purpose of the SynthNet Evaluation Pipeline is to evaluate the retrieval performance of trained feature extractors.

# Getting Started

## Linux
1. Open terminal in project root directory
2. Create [pyenv](https://github.com/pyenv/pyenv) environment using `python 3.9`
 ```bash
pyenv install 3.9
pyenv virtualenv synthnet-eval-pipeline
```
4. Activate the pyenv environment and install dependencies from from [requirements.txt](./requirements.txt)
 ```bash
pyenv shell synthnet-eval-pipeline
pip install -r requirements.txt
```

# Usage




## Data Preparation
Image dataset directory structure must be followed as some information is pulled out of it during a run (class names, entity names etc.).

Image data must follow the schema `{dataset}/{split}/images/{class}/{entity}`. <br/>
Example: `{ModelNet10}/{train}/images/{horse}/{horse_001}`.


## Run Config
Example run configuration file.
```json
{
    "run_name": "modelnet10_vitin21k", // Name of the run to start
    "data_train_roots": [
        "./data/modelnet10/train" // Root directory of train dataset (search index)
    ],
    "data_test_roots": [
        "./data/modelnet10/test" // Root directory of test dataset (query images)
    ],
    "out_root_dir": "./out", // Root director for all outputs
    "seed": 42, // Random seed for reproducability
    "splits": [
        {
            "split_name": "traintest_full" // Split option to use. (see conf/_enums.py)
        }
    ],
    "feature_extractors": [
        {
            "feature_extractor_name": "vit_in21k", // Feature extractor to use (see conf/_enums.py)
            "feature_dims": 768 // Feature vector length
        }
    ]
}
```

## Build Search Index & Perform Retrieval
To perform a retrieval run you have to prepare your data create a run config json file first. Afterwards you can navigate to the project directory and start a pipeline run the following way:
```bash
python run_retrieval.py --run_config_path ./run_configs/my_run_config.json
```


---
## Evaluate Retrieval Performance
To measure retrieval performance use the `evaluation_metrics.ipynb` jupyter notebook and use the `visualization.ipynb` to create useful visualizations as confusion matrix or TSNE scatter plot.


# Outputs
* (optional) feature Vectors extracted for all images.
* image indices in FAISS search index mapped to image filename
* retrieval results for all query images
* calculated eval metrics

## Metrics
* Accuracy at 1, 5, 10
* Mean Average Precision (mAP) at 1, 5, 10, 100, R (R = Maximale Anzahl der relevanten/korrekten Ergebnisse)
* Average Normalized Discounted Cumulative Gain (NDCG) at 5, 10 ,100, N (N = Anzahl der Retrievals)
* Mean Reciprocal Rank (MRR)


