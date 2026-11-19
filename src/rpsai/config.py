'''Typed configuration loaded from config/config.yaml.'''

import logging
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path('config/config.yaml')
SPLITS = ('train', 'val', 'test')
GESTURE_CLASSES = ('rock', 'paper', 'scissors', 'other')

# Third-party loggers that print lines unrelated to the pipeline stages
_NOISY_LOGGERS = ('matplotlib', 'PIL')


class StrictModel(BaseModel):
    '''Base class that rejects unknown keys so typos in the YAML file fail fast.'''

    model_config = ConfigDict(extra='forbid', frozen=True)


class DataConfig(StrictModel):
    '''Where downloads and intermediate files are stored.'''

    raw_dir: Path
    processed_dir: Path


class Download(StrictModel):
    '''A file to download, pinned by its checksum.'''

    url: str = Field(pattern=r'^https://')
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class DownloadConfig(StrictModel):
    '''The two datasets.'''

    hagrid: Download
    human_games: Download


class GestureConfig(StrictModel):
    '''How the gesture reader is trained.'''

    seed: int
    classes: dict[str, tuple[str, ...]]
    hands_per_class: dict[str, int]
    stretch_range: tuple[float, float]
    stretched_copies: int = Field(ge=1)
    hidden_layers: tuple[int, ...] = Field(min_length=1)
    l2_penalty: float = Field(ge=0)
    max_epochs: int = Field(ge=1)
    max_wrong_move_rate: float = Field(gt=0, lt=1)

    @model_validator(mode='after')
    def _check(self) -> Self:
        if tuple(self.classes) != GESTURE_CLASSES:
            raise ValueError(f'classes must be {", ".join(GESTURE_CLASSES)} in that order')
        labels = [label for members in self.classes.values() for label in members]
        if len(labels) != len(set(labels)):
            raise ValueError('a HaGRID label may belong to only one class')
        if sorted(self.hands_per_class) != sorted(SPLITS):
            raise ValueError(f'hands_per_class needs exactly {", ".join(SPLITS)}')
        low, high = self.stretch_range
        if not 0 < low <= 1 <= high:
            raise ValueError('stretch_range must contain 1')
        return self

    def class_of(self) -> dict[str, int]:
        '''Map each HaGRID label to its class index.'''
        return {
            label: index for index, members in enumerate(self.classes.values()) for label in members
        }


class PredictorGrid(StrictModel):
    '''Candidate values for each tuned predictor setting.'''

    decay: tuple[float, ...] = Field(min_length=1)
    prior: tuple[float, ...] = Field(min_length=1)
    learning_rate: tuple[float, ...] = Field(min_length=1)
    score_decay: tuple[float, ...] = Field(min_length=1)


class PredictorConfig(StrictModel):
    '''How the move predictor is tuned and evaluated.'''

    seed: int
    tuning_fraction: float = Field(gt=0, lt=1)
    pattern_length: int = Field(ge=1)
    grid: PredictorGrid
    margins: tuple[float, ...] = Field(min_length=1)
    bootstrap_samples: int = Field(ge=100)


class OutputConfig(StrictModel):
    '''Where results go.'''

    report_dir: Path
    web_model_dir: Path
    fixture_dir: Path


class AppConfig(StrictModel):
    '''The whole configuration file.'''

    data: DataConfig
    downloads: DownloadConfig
    gestures: GestureConfig
    predictor: PredictorConfig
    outputs: OutputConfig


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    '''Read and validate the YAML configuration.'''
    with path.open(encoding='utf-8') as handle:
        raw: Any = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)


def configure_logging(level: int = logging.INFO) -> None:
    '''Print plain progress lines from the pipeline and quieten chatty libraries.'''
    logging.basicConfig(level=level, format='%(message)s', force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
