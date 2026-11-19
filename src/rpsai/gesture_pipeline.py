'''Training and evaluating the gesture reader, from HaGRID landmarks to the model the game loads.'''

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from rpsai.config import GESTURE_CLASSES, SPLITS, AppConfig
from rpsai.gestures import (
    UNSURE,
    GestureMetrics,
    GestureNetwork,
    choose_threshold,
    decide,
    evaluate,
    one_hot,
    rule_based,
    train_logistic,
    train_network,
    wrong_move_rate,
)
from rpsai.hagrid import GestureSample, LandmarkTable, draw_sample, extract_landmarks
from rpsai.landmarks import features, random_stretches

logger = logging.getLogger(__name__)

LANDMARK_CACHE = 'hagrid_landmarks.npz'
# Vertical stretches applied to test hands to check that readers ignore the image shape
STRETCHES = (0.5, 0.63, 0.79, 1.0, 1.26, 1.59, 2.0)
THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.30, 1.0, 0.05), 2))
# Test hands drawn for each label that belongs to no class
LEFT_OUT_HANDS = 1000
LEFT_OUT = 'left out'

Reader = Callable[[GestureSample, float], np.ndarray]


@dataclass(frozen=True)
class ReaderResult:
    '''Test results of one way of reading hands.'''

    name: str
    metrics: GestureMetrics
    # The finger-counting rule makes hard decisions, so a threshold does not apply to it
    thresholded: bool
    # Accuracy on test hands stretched by each factor in STRETCHES
    stretch_accuracy: tuple[float, ...]


@dataclass(frozen=True)
class LabelReading:
    '''What the shipped reader makes of the test hands of one HaGRID label.'''

    label: str
    gesture_class: str
    hands: int
    # Shares read as rock, paper, scissors, and other, then the share asked to show again
    shares: tuple[float, ...]


@dataclass(frozen=True)
class GestureResults:
    '''Everything the report needs about the gesture reader.'''

    hands: dict[str, int]
    people: dict[str, int]
    hidden_layers: tuple[int, ...]
    parameters: int
    stretches: tuple[float, ...]
    readers: tuple[ReaderResult, ...]
    # Threshold, wrong-move rate, and share of moves asked again, for the shipped reader on test
    threshold_curve: tuple[tuple[float, float, float], ...]
    labels: tuple[LabelReading, ...]


def load_landmarks(config: AppConfig) -> LandmarkTable:
    '''Load the cached landmark table, extracting it from the annotations on first use.'''
    cache = config.data.processed_dir / LANDMARK_CACHE
    if cache.exists():
        return LandmarkTable.load(cache)
    logger.info('Extracting landmarks from the HaGRID annotations')
    table = extract_landmarks(config.data.raw_dir / config.downloads.hagrid.file)
    table.save(cache)
    return table


def train_gesture_reader(
    config: AppConfig, table: LandmarkTable
) -> tuple[GestureNetwork, GestureResults]:
    '''Train the three readers, choose the threshold on validation hands, and test them.'''
    settings = config.gestures
    class_of = settings.class_of()
    samples = {
        split: draw_sample(table, split, class_of, settings.hands_per_class[split], settings.seed)
        for split in SPLITS
    }
    for split, sample in samples.items():
        logger.info(
            'Drew %s %s hands from %s people',
            f'{len(sample):,}',
            split,
            f'{len(np.unique(sample.users)):,}',
        )

    train = samples['train']
    copies = settings.stretched_copies
    points = np.concatenate([train.points] * copies)
    targets = np.concatenate([train.targets] * copies)
    inputs = features(points, random_stretches(len(points), settings.stretch_range, settings.seed))

    logger.info('Training logistic regression')
    logistic = train_logistic(inputs, targets).rounded()
    logger.info('Training the neural network')
    network = train_network(
        inputs,
        targets,
        settings.hidden_layers,
        settings.l2_penalty,
        settings.max_epochs,
        settings.seed,
    ).rounded()

    readers: tuple[tuple[str, Reader, bool], ...] = (
        ('Finger-counting rule', _read_rules, False),
        ('Logistic regression', partial(_read, logistic), True),
        ('Neural network', partial(_read, network), True),
    )
    validation, test = samples['val'], samples['test']
    results = []
    for name, read, thresholded in readers:
        threshold = (
            choose_threshold(
                read(validation, 1.0), validation.targets, settings.max_wrong_move_rate
            )
            if thresholded
            else 0.0
        )
        metrics = evaluate(read(test, 1.0), test.targets, threshold)
        stretch_accuracy = tuple(
            float(np.mean(read(test, stretch).argmax(axis=-1) == test.targets))
            for stretch in STRETCHES
        )
        results.append(ReaderResult(name, metrics, thresholded, stretch_accuracy))
        logger.info('%s: test accuracy %.2f%%', name, 100 * metrics.accuracy)

    # The neural network is the last reader and the one the game uses
    shipped = network.with_threshold(results[-1].metrics.threshold)
    probabilities = _read(shipped, test)
    moves = np.isin(test.targets, (0, 1, 2))
    curve = tuple(
        (
            threshold,
            wrong_move_rate(decide(probabilities, threshold), test.targets),
            float(np.mean(~np.isin(decide(probabilities, threshold)[moves], (0, 1, 2)))),
        )
        for threshold in THRESHOLDS
    )
    return shipped, GestureResults(
        hands={split: len(sample) for split, sample in samples.items()},
        people={split: len(np.unique(sample.users)) for split, sample in samples.items()},
        hidden_layers=settings.hidden_layers,
        parameters=sum(
            w.size + b.size for w, b in zip(shipped.weights, shipped.biases, strict=True)
        ),
        stretches=STRETCHES,
        readers=tuple(results),
        threshold_curve=curve,
        labels=_label_readings(config, table, shipped, test),
    )


def _read(network: GestureNetwork, sample: GestureSample, stretch: float = 1.0) -> np.ndarray:
    return network.probabilities(features(sample.points, stretch))


def _read_rules(sample: GestureSample, stretch: float = 1.0) -> np.ndarray:
    return one_hot(rule_based(sample.points, stretch))


def _label_readings(
    config: AppConfig, table: LandmarkTable, network: GestureNetwork, test: GestureSample
) -> tuple[LabelReading, ...]:
    '''Break the shipped reader's test decisions down by HaGRID label, including left-out ones.'''
    class_of = config.gestures.class_of()
    rows = []
    for label in sorted(class_of, key=lambda name: (class_of[name], name)):
        mask = test.labels == label
        if mask.any():
            rows.append((label, GESTURE_CLASSES[class_of[label]], test.points[mask]))
    rng = np.random.default_rng(config.gestures.seed)
    in_test = table.split_ids == SPLITS.index('test')
    for label in sorted(set(table.label_names) - set(class_of)):
        found = np.flatnonzero(in_test & (table.label_ids == table.label_names.index(label)))
        chosen = rng.permutation(found)[:LEFT_OUT_HANDS]
        if len(chosen):
            rows.append((label, LEFT_OUT, table.points[np.sort(chosen)]))
    readings = []
    for label, gesture_class, points in rows:
        decisions = decide(network.probabilities(features(points)), network.threshold)
        shares = [float(np.mean(decisions == index)) for index in range(len(GESTURE_CLASSES))]
        shares.append(float(np.mean(decisions == UNSURE)))
        readings.append(LabelReading(label, gesture_class, len(points), tuple(shares)))
    return tuple(readings)
