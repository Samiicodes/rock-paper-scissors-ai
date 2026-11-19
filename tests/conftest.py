from pathlib import Path

import numpy as np
import pytest
from synthetic import hagrid_archive, human_games_archive

from rpsai.config import AppConfig, load_config
from rpsai.predictors import PredictorSettings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> PredictorSettings:
    return PredictorSettings(
        decay=0.9, prior=1.0, learning_rate=1.0, score_decay=0.95, pattern_length=6, margin=0.0
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def repo_config() -> AppConfig:
    return load_config(REPO_ROOT / 'config' / 'config.yaml')


@pytest.fixture
def small_config(tmp_path: Path, repo_config: AppConfig) -> AppConfig:
    '''The repository configuration scaled down to synthetic data in a temporary directory.'''
    raw = tmp_path / 'data' / 'raw'
    raw.mkdir(parents=True)
    hagrid_archive(raw / repo_config.downloads.hagrid.file, hands_per_label=600)
    human_games_archive(raw / repo_config.downloads.human_games.file)
    gestures = repo_config.gestures.model_copy(
        update={
            'classes': {
                'rock': ('fist',),
                'paper': ('palm',),
                'scissors': ('peace',),
                'other': ('one', 'no_gesture'),
            },
            'hands_per_class': {'train': 500, 'val': 100, 'test': 100},
            'stretched_copies': 2,
            'hidden_layers': (16,),
            'max_epochs': 60,
            'max_wrong_move_rate': 0.05,
        }
    )
    grid = repo_config.predictor.grid.model_copy(
        update={
            'decay': (0.9,),
            'prior': (1.0,),
            'learning_rate': (1.0, 2.0),
            'score_decay': (0.95,),
        }
    )
    predictor = repo_config.predictor.model_copy(
        update={'grid': grid, 'margins': (0.0, 0.05), 'bootstrap_samples': 100}
    )
    return repo_config.model_copy(
        update={
            'data': repo_config.data.model_copy(
                update={'raw_dir': raw, 'processed_dir': tmp_path / 'data' / 'processed'}
            ),
            'gestures': gestures,
            'predictor': predictor,
            'outputs': repo_config.outputs.model_copy(
                update={
                    'report_dir': tmp_path / 'reports',
                    'web_model_dir': tmp_path / 'web' / 'models',
                    'fixture_dir': tmp_path / 'web' / 'fixtures',
                }
            ),
        }
    )
