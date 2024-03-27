""" Contains enums for run_config validation """

from enum import Enum
from typing import List


class CustomEnum(Enum):

    @classmethod
    def values_to_list(cls) -> List[str]:
        """Return all enum values as list of strings

        Returns:
            l (list[str]): List containing the enum values as strings
        """
        return [str(item) for item in cls.__members__]

    def __str__(self) -> str:
        return self.value


class FeatureExtractors(CustomEnum):
    """ Class contains all available feature extractors """
    vit_in21k = 'vit_in21k'
    vit_in21k_finetuned = 'vit_in21k_finetuned'
    convnext_s = 'convnext_s'
    dino_vitb16_in1k = 'dino_vitb16_in1k'


class Splits(CustomEnum):
    """ Class contains all available split configurations """
    traintest_full = 'traintest_full'
    # syn2real = 'syn2real'
    # syn2real_fitted = 'syn2real_fitted'


