'''The whole pipeline: train, evaluate, and write the game's files and the report.'''

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from rpsai import plotting
from rpsai.config import AppConfig
from rpsai.export import (
    GESTURE_FIXTURE_FILE,
    GESTURE_MODEL_FILE,
    PREDICTOR_FILE,
    PREDICTOR_FIXTURE_FILE,
    gesture_fixture,
    gesture_model_payload,
    predictor_fixture,
    predictor_payload,
    write_json,
)
from rpsai.gesture_pipeline import GestureResults, load_landmarks, train_gesture_reader
from rpsai.predictor_pipeline import PredictorResults, run_predictor_pipeline
from rpsai.reporting import FIGURE_DIR_NAME, METRICS_FILE, REPORT_FILE, write_metrics, write_report

logger = logging.getLogger(__name__)

DOCS_DIR = Path('docs')


@dataclass(frozen=True)
class PipelineOutcome:
    '''The results and where they were written.'''

    gestures: GestureResults
    predictor: PredictorResults
    report_path: Path


def make_figures(
    gestures: GestureResults, predictor: PredictorResults, report_dir: Path
) -> list[tuple[str, Path]]:
    '''Draw every chart and return titles with paths relative to the report directory.'''
    figure_dir = report_dir / FIGURE_DIR_NAME
    network = gestures.readers[-1]
    drawn = [
        (
            'AI strategies against the test players',
            plotting.plot_strategies(predictor, figure_dir / 'strategies.png'),
        ),
        (
            'Learning a player during a game',
            plotting.plot_learning(predictor, figure_dir / 'learning.png'),
        ),
        (
            'What the combined forecast relies on',
            plotting.plot_expert_weights(predictor, figure_dir / 'expert_weights.png'),
        ),
        (
            'Three ways of reading hands',
            plotting.plot_readers(gestures, figure_dir / 'readers.png'),
        ),
        (
            'Reading every HaGRID gesture',
            plotting.plot_labels(gestures, figure_dir / 'labels.png'),
        ),
        (
            'Robustness to image shape',
            plotting.plot_stretch(gestures, figure_dir / 'stretch.png'),
        ),
        (
            'Neural network: hand shown against hand read',
            plotting.plot_confusion(network.metrics.confusion, figure_dir / 'confusion.png'),
        ),
    ]
    return [(title, path.relative_to(report_dir)) for title, path in drawn]


def run_pipeline(config: AppConfig, publish_docs: bool = False) -> PipelineOutcome:
    '''Train the gesture reader, tune the predictor, export the game files, and write the report.'''
    table = load_landmarks(config)
    network, gestures = train_gesture_reader(config, table)
    predictor = run_predictor_pipeline(config)

    models, fixtures = config.outputs.web_model_dir, config.outputs.fixture_dir
    write_json(gesture_model_payload(network), models / GESTURE_MODEL_FILE)
    write_json(predictor_payload(predictor.settings), models / PREDICTOR_FILE, indent=2)
    write_json(gesture_fixture(network), fixtures / GESTURE_FIXTURE_FILE)
    write_json(predictor_fixture(predictor.settings), fixtures / PREDICTOR_FIXTURE_FILE)
    logger.info('Wrote the game files to %s', models.as_posix())

    report_dir = config.outputs.report_dir
    figures = make_figures(gestures, predictor, report_dir)
    write_metrics(gestures, predictor, report_dir / METRICS_FILE)
    report_path = write_report(gestures, predictor, figures, report_dir / REPORT_FILE)
    logger.info('Report written to %s', report_path.as_posix())
    if publish_docs:
        shutil.copy2(report_path, DOCS_DIR / REPORT_FILE)
        shutil.copytree(
            report_dir / FIGURE_DIR_NAME, DOCS_DIR / FIGURE_DIR_NAME, dirs_exist_ok=True
        )
        logger.info('Published report to %s', DOCS_DIR.as_posix())
    return PipelineOutcome(gestures, predictor, report_path)
