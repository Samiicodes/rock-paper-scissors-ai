'''The gesture reader: turns hand landmarks into rock, paper, scissors, or no move.'''

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from rpsai.config import GESTURE_CLASSES
from rpsai.landmarks import FEATURE_COUNT, extended_fingers

ROCK, PAPER, SCISSORS, OTHER = range(4)
MOVE_CLASSES = (ROCK, PAPER, SCISSORS)
# A decision of UNSURE means the game asks the player to show their hand again
UNSURE = -1
EXPORT_FORMAT = 1
SIGNIFICANT_DIGITS = 6


def rule_based(points: np.ndarray, aspect: float | np.ndarray = 1.0) -> np.ndarray:
    '''Classify hands by counting straight fingers, the baseline for the trained readers.

    No straight finger is rock, all four is paper, and the index and middle fingers alone are
    scissors. Anything else is other.
    '''
    fingers = extended_fingers(points, aspect)
    count = fingers.sum(axis=-1)
    result = np.full(count.shape, OTHER)
    result[count == 0] = ROCK
    result[count == 4] = PAPER
    scissors = fingers[..., 0] & fingers[..., 1] & ~fingers[..., 2] & ~fingers[..., 3]
    result[scissors] = SCISSORS
    return result


def one_hot(classes: np.ndarray) -> np.ndarray:
    '''Turn hard class decisions into probabilities of 0 and 1.'''
    return np.asarray(np.eye(len(GESTURE_CLASSES))[classes])


@dataclass(frozen=True)
class GestureNetwork:
    '''A multilayer perceptron with ReLU hidden layers and a softmax output.

    With no hidden layers it is multinomial logistic regression.
    '''

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.biases) or not self.weights:
            raise ValueError('one bias vector is needed per weight matrix')
        if self.weights[0].shape[0] != FEATURE_COUNT:
            raise ValueError(f'the first layer must take {FEATURE_COUNT} inputs')
        if self.weights[-1].shape[1] != len(GESTURE_CLASSES):
            raise ValueError(f'the last layer must have {len(GESTURE_CLASSES)} outputs')

    def probabilities(self, inputs: np.ndarray) -> np.ndarray:
        '''Return the probability of each class for each row of features.'''
        activation = np.asarray(inputs, dtype=np.float64)
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            activation = activation @ weight + bias
            if index < last:
                activation = np.maximum(activation, 0.0)
        shifted = np.exp(activation - activation.max(axis=-1, keepdims=True))
        return np.asarray(shifted / shifted.sum(axis=-1, keepdims=True))

    def with_threshold(self, threshold: float) -> Self:
        '''Return the same network with a new acceptance threshold.'''
        return type(self)(self.weights, self.biases, threshold)

    def rounded(self) -> Self:
        '''Round every parameter as the export does, so results describe the shipped model.'''
        return type(self).from_json(self.to_json())

    def to_json(self) -> dict[str, Any]:
        '''Describe the network for the web game.'''
        return {
            'format': EXPORT_FORMAT,
            'classes': list(GESTURE_CLASSES),
            'threshold': round(self.threshold, 4),
            'layers': [
                {'weights': _round(weight), 'bias': _round(bias)}
                for weight, bias in zip(self.weights, self.biases, strict=True)
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        '''Rebuild a network written by to_json.'''
        if payload.get('format') != EXPORT_FORMAT:
            raise ValueError('unsupported gesture model format')
        if tuple(payload['classes']) != GESTURE_CLASSES:
            raise ValueError('gesture model classes do not match')
        layers = payload['layers']
        return cls(
            weights=tuple(np.array(layer['weights'], dtype=np.float64) for layer in layers),
            biases=tuple(np.array(layer['bias'], dtype=np.float64) for layer in layers),
            threshold=float(payload['threshold']),
        )


def _round(values: np.ndarray) -> Any:
    return np.vectorize(lambda value: float(f'{value:.{SIGNIFICANT_DIGITS}g}'))(values).tolist()


def train_network(
    inputs: np.ndarray,
    targets: np.ndarray,
    hidden_layers: Sequence[int],
    l2_penalty: float,
    max_epochs: int,
    seed: int,
) -> GestureNetwork:
    '''Train a multilayer perceptron with Adam and early stopping on a held-out tenth.'''
    model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        alpha=l2_penalty,
        max_iter=max_epochs,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(inputs, targets)
    _check_classes(model.classes_)
    return GestureNetwork(
        weights=tuple(np.asarray(weight) for weight in model.coefs_),
        biases=tuple(np.asarray(bias) for bias in model.intercepts_),
    )


def train_logistic(inputs: np.ndarray, targets: np.ndarray) -> GestureNetwork:
    '''Train multinomial logistic regression, the linear baseline.'''
    model = LogisticRegression(max_iter=2000)
    model.fit(inputs, targets)
    _check_classes(model.classes_)
    return GestureNetwork(weights=(model.coef_.T.copy(),), biases=(model.intercept_.copy(),))


def _check_classes(classes: np.ndarray) -> None:
    if list(classes) != list(range(len(GESTURE_CLASSES))):
        raise ValueError('training data must contain every class')


def decide(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    '''Return the most likely class, or UNSURE when it is less likely than the threshold.'''
    best = probabilities.argmax(axis=-1)
    return np.where(probabilities.max(axis=-1) >= threshold, best, UNSURE)


def wrong_move_rate(decisions: np.ndarray, targets: np.ndarray) -> float:
    '''Share of hands that the game would play as a move the player did not show.

    This counts a move read as a different move, and a non-move read as a move.
    '''
    played = np.isin(decisions, MOVE_CLASSES)
    return float(np.mean(played & (decisions != targets)))


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray, max_rate: float) -> float:
    '''Return the lowest threshold whose wrong-move rate is at most max_rate.'''
    for threshold in np.round(np.arange(0.30, 1.0, 0.01), 2):
        if wrong_move_rate(decide(probabilities, threshold), targets) <= max_rate:
            return float(threshold)
    return 0.99


@dataclass(frozen=True)
class GestureMetrics:
    '''How well a classifier reads hands, with and without the game's threshold.'''

    accuracy: float
    macro_f1: float
    confusion: tuple[tuple[int, ...], ...]
    # With the threshold applied, as in the game
    threshold: float
    moves_read_correctly: float
    moves_asked_again: float
    wrong_move_rate: float
    others_played_as_moves: float


def evaluate(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> GestureMetrics:
    '''Score class probabilities against the true classes.'''
    predicted = probabilities.argmax(axis=-1)
    classes = len(GESTURE_CLASSES)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    decisions = decide(probabilities, threshold)
    moves = np.isin(targets, MOVE_CLASSES)
    others = targets == OTHER
    return GestureMetrics(
        accuracy=float(np.mean(predicted == targets)),
        macro_f1=float(f1_score(targets, predicted, average='macro')),
        confusion=tuple(tuple(int(value) for value in row) for row in confusion),
        threshold=threshold,
        moves_read_correctly=float(np.mean(decisions[moves] == targets[moves])),
        moves_asked_again=float(np.mean(~np.isin(decisions[moves], MOVE_CLASSES))),
        wrong_move_rate=wrong_move_rate(decisions, targets),
        others_played_as_moves=float(np.mean(np.isin(decisions[others], MOVE_CLASSES))),
    )
