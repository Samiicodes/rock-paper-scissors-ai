import json
from pathlib import Path

import pytest

from rpsai.config import AppConfig
from rpsai.export import GESTURE_FIXTURE_FILE, GESTURE_MODEL_FILE, PREDICTOR_FILE
from rpsai.gesture_pipeline import LANDMARK_CACHE
from rpsai.pipeline import run_pipeline
from rpsai.reporting import interval, percent
from rpsai.simulation import Interval


def test_pipeline_runs_end_to_end(
    small_config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'docs').mkdir()
    outcome = run_pipeline(small_config, publish_docs=True)

    gestures, predictor = outcome.gestures, outcome.predictor
    assert [reader.name for reader in gestures.readers] == [
        'Finger-counting rule',
        'Logistic regression',
        'Neural network',
    ]
    assert gestures.readers[-1].metrics.accuracy > 0.8
    assert {reading.gesture_class for reading in gestures.labels} >= {'rock', 'left out'}
    assert predictor.skipped_players == (7,)
    assert predictor.strategies[-1].name == 'All predictors combined'
    # The synthetic players cycle through the moves, so the AI should beat chance comfortably
    assert predictor.strategies[-1].win_rate.value > 0.45

    outputs = small_config.outputs
    model = json.loads((outputs.web_model_dir / GESTURE_MODEL_FILE).read_text())
    assert 'HaGRID' in model['attribution']
    assert json.loads((outputs.web_model_dir / PREDICTOR_FILE).read_text())['format'] == 1
    assert (outputs.fixture_dir / GESTURE_FIXTURE_FILE).exists()
    assert (small_config.data.processed_dir / LANDMARK_CACHE).exists()

    report = outcome.report_path.read_text(encoding='utf-8')
    for heading in ('## Gesture reader', '## Move predictor', '## Figures'):
        assert heading in report
    figures = list((outputs.report_dir / 'figures').glob('*.png'))
    assert len(figures) == 7
    assert (tmp_path / 'docs' / 'evaluation_report.md').read_text(encoding='utf-8') == report
    assert len(list((tmp_path / 'docs' / 'figures').glob('*.png'))) == 7

    # A second run reuses the cached landmarks instead of reading the annotations again
    (small_config.data.raw_dir / small_config.downloads.hagrid.file).unlink()
    assert run_pipeline(small_config).gestures.hands == gestures.hands


def test_report_formatting() -> None:
    assert percent(0.4233) == '42.3%'
    assert percent(0.5, 2) == '50.00%'
    assert interval(Interval(0.42, 0.40, 0.44)) == '42.0% (40.0 to 44.0)'
    assert interval(Interval(0.15, -0.01, 0.2), signed=True) == '+15.0% (-1.0 to +20.0)'
