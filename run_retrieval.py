""" Script used to run the evaluation pipeline"""
from types import SimpleNamespace
import logging
import sys
import os
import json
import shutil
from jsonschema import ValidationError
import click

from conf import paths
from conf._schemas import create_validator, CONFIG_SCHEMA
from modules.create_dataset_splits import create_splits
from modules.build_index import build_index
from modules.retrieval import get_retrieval_results
# from modules.calc_metrics import calc_metrics

RUN_CONFIG_NAME_OUT = "run_config.json"
log = logging.getLogger("STATUS")
logging.basicConfig(level=logging.INFO)


def load_config(**kwargs) -> SimpleNamespace:
    """Load the config file from run_config_path parsed from given kwargs

    Args:
        **kwargs: cli keyword arguments

    Returns:
        config (dict): loaded config as dict representation
    """
    args = SimpleNamespace(**kwargs)

    # Load config file
    with open(args.run_config_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    log.info(f"START EVALUATION WITH CONFIG \n {json.dumps(config, indent=2)}")

    validator = create_validator(CONFIG_SCHEMA)
    try:
        validator.validate(config)
    except ValidationError as e:
        sys.exit(f"Error validating config file at {paths.base_run_config_path}: {e}")

    config = SimpleNamespace(**config)

    # Initialize global paths object
    paths.init_base_paths(out_dir=config.out_root_dir,
                          run_name=config.run_name,
                          run_config_path=args.run_config_path,
                          data_train_roots=config.data_train_roots,
                          data_test_roots=config.data_test_roots)

    # Make dicts for each split and feature extractor
    # Parse splits/feature extractors
    for fe in config.feature_extractors:
        for split in config.splits:
            paths.add_fe_split_pair(fe["feature_extractor_name"], split["split_name"])

    # Initialize out directory
    os.makedirs(paths.base_run_dir, exist_ok=True)
    shutil.copyfile(paths.base_run_config_path, paths.base_out_run_config_path)

    return config


# TODO: Add click options for each field in run_config. Then use config params as defaults.
@click.command()
@click.option(
    '--run_config_path',
    '-c',
    help="Path to the run config json file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
# @click.option(
#     '--out_root_dir',
#     '-o',
#     help='Name of the directory, where the results shall be stored',
#     type=click.Path(exists=False),
#     show_default=True,
#     default='./out',
# )
def main(**kwargs):

    # Parse click parameters and load config
    config = load_config(**kwargs)

    log.info('CREATING DATASET SPLITS')
    create_splits(seed=config.seed)
    print(config)
    for fe_dict in config.feature_extractors:
        for split_dict in config.splits:  # pylint: disable=unused-variable
            log.info('- ' * 20)
            log.info(f'RUNNING {fe_dict["feature_extractor_name"]} {split_dict["split_name"]} ')
            # Build index
            log.info('BUILDING SEARCH INDEX')

            fe_path = fe_dict["feature_extractor_path"] if "feature_extractor_path" in fe_dict else None
            fe_name = fe_dict["feature_extractor_name"]
            fe_dims = fe_dict["feature_dims"]
            build_index(
                dataset_split=split_dict["split_name"],
                feature_extractor_name=fe_name,
                feature_extractor_path=fe_path,
                feature_extractor_dims=fe_dims,
                feature_map=None,
            )
            # Get retrievals
            log.info('PERFORM RETRIEVAL QUERIES')
            retrieval_df = get_retrieval_results(split_dict["split_name"], fe_dict["feature_extractor_name"])
            # Calculate metrics
            # log.info('CALC EVAL METRICS')
            # calc_metrics(retrieval_df=retrieval_df, split=split, feature_extractor=fe)


if __name__ == '__main__':
    main()
