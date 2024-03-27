""" Contains logic and schemes for json validation """

import os
from jsonschema import Draft202012Validator, FormatChecker, Validator

from ._enums import Splits, FeatureExtractors


def is_dir(instance: str, dir_level: int = 0) -> bool:
    """Check if the given instance is a valid directory at the level indicated by dir_level

    Args:
        instance (str): Path to be evaluated
        dir_level (int): Indicates whether to check if the last (0) or second to last 
        (1) ... item of the path is a directory
    
    Returns:
        is_directory (bool): True if instance is valid directory, False otherwise
    """
    if not isinstance(instance, str):
        return False
    path_split = instance.split(os.sep)
    end_of_path_idx = len(path_split) - dir_level
    if end_of_path_idx <= 0:
        return False
    path_combined = f'{os.sep}'.join(path_split[:end_of_path_idx])
    return os.path.isdir(path_combined)


# Initialize FormatChecker format "directory" format,
#   used to assert if path is an existing directory
format_checker = FormatChecker()


@format_checker.checks("directory", AssertionError)
def is_last_elem_directory(instance: str) -> bool:
    return is_dir(instance, dir_level=0)


@format_checker.checks("one_valid_directory_in_path")
def is_one_valid_dir_in_path(instance: str) -> bool:
    if not isinstance(instance, str):
        return False
    splitted = instance.split(os.sep)
    for idx in range(len(splitted)):
        is_valid = is_dir(instance, dir_level=idx)
        if is_valid:
            return True
    return False


# Schema used to validate the run_config. Update, if new parameters are needed
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "run_name": {
            "type": "string"
        },
        "data_train_roots": {
            "type": "array",
            "items": {
                "type": "string",
                "format": "directory"
            }
        },
        "data_test_roots": {
            "type": "array",
            "items": {
                "type": "string",
                "format": "directory"
            }
        },
        "out_root_dir": {
            "type": "string",
            "format": "one_valid_directory_in_path"
        },
        "seed": {
            "type": "number"
        },
        "splits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "split_name": {
                        "enum": Splits.values_to_list()
                    }
                },
                "required": ["split_name"]
            },
            "minItems": 1,
            "unique_items": True
        },
        "feature_extractors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature_extractor_name": {
                        "enum": FeatureExtractors.values_to_list()
                    },
                    "feature_extractor_path": {
                        "type": "string",
                        "format": "one_valid_directory_in_path"
                    },
                    "feature_dims": {
                        "type": "number"
                    }
                },
                "required": ["feature_extractor_name", "feature_dims"]
            },
            "minItems": 1,
            "unique_items": True
        }
    },
    "required": [
        "run_name",
        "data_train_roots",
        "data_test_roots",
        "out_root_dir",
        "seed",
        "splits",
        "feature_extractors",
    ]
}


def create_validator(schema: dict) -> Validator:
    """Used to create a validator with the given Schema

    Args:
        schema (dict): Schema used in the Validator to validate json data, will be checked if valid

    Returns:
        validator (Validator): the Validator created
    """
    vali = Draft202012Validator(schema=schema, format_checker=format_checker)
    vali.check_schema(schema)
    return vali
