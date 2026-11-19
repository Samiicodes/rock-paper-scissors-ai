from pathlib import Path

import numpy as np
import pytest
from synthetic import FINGERS, hagrid_archive, hand_in_hand_coordinates, make_hand

from rpsai.gestures import (
    OTHER,
    PAPER,
    ROCK,
    SCISSORS,
    UNSURE,
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
from rpsai.hagrid import (
    LandmarkTable,
    _spread,
    annotation_files,
    draw_sample,
    extract_landmarks,
)
from rpsai.landmarks import (
    FEATURE_COUNT,
    canonicalise,
    extended_fingers,
    features,
    random_stretches,
)

CLASS_OF = {'fist': 0, 'palm': 1, 'peace': 2, 'one': 3, 'no_gesture': 3}


def labelled_hands(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    names = ['fist', 'palm', 'peace', 'one']
    targets = np.arange(count) % 4
    points = np.array([make_hand(FINGERS[names[target]], rng) for target in targets])
    return points, targets


def test_canonicalise_fixes_position_size_and_direction(rng: np.random.Generator) -> None:
    canonical = canonicalise(make_hand(FINGERS['palm'], rng))
    assert canonical[0] == pytest.approx([0.0, 0.0])
    assert canonical[9] == pytest.approx([0.0, -1.0])
    assert canonical[5, 0] > canonical[17, 0]


def test_canonicalise_ignores_rotation_scale_and_mirroring() -> None:
    hand = hand_in_hand_coordinates(FINGERS['peace'])
    angle = 1.1
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = hand @ rotation.T * 0.2 + 0.4
    mirrored = moved * np.array([-1.0, 1.0])
    assert canonicalise(moved) == pytest.approx(canonicalise(hand))
    assert canonicalise(mirrored) == pytest.approx(canonicalise(hand))


def test_canonicalise_undoes_the_image_aspect_ratio() -> None:
    hand = hand_in_hand_coordinates(FINGERS['fist']) * 0.1 + 0.5
    squashed = hand * np.array([1.0, 0.5])
    assert canonicalise(squashed, aspect=2.0) == pytest.approx(canonicalise(hand))


def test_canonicalise_accepts_one_aspect_per_hand(rng: np.random.Generator) -> None:
    hands = np.array([make_hand(FINGERS['palm'], rng) for _ in range(3)])
    together = canonicalise(hands, np.array([0.5, 1.0, 2.0]))
    assert together[2] == pytest.approx(canonicalise(hands[2], 2.0))


def test_features_drop_the_wrist(rng: np.random.Generator) -> None:
    assert features(make_hand(FINGERS['fist'], rng)).shape == (FEATURE_COUNT,)


def test_random_stretches_stay_in_range_and_centre_on_one() -> None:
    stretches = random_stretches(4000, (0.5, 2.0), seed=0)
    assert stretches.min() >= 0.5
    assert stretches.max() <= 2.0
    assert np.median(stretches) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(
    ('name', 'expected'), [('fist', ROCK), ('palm', PAPER), ('peace', SCISSORS), ('one', OTHER)]
)
def test_rule_based_counts_fingers(rng: np.random.Generator, name: str, expected: int) -> None:
    hand = make_hand(FINGERS[name], rng)
    assert list(extended_fingers(hand)) == list(FINGERS[name])
    assert rule_based(hand) == expected


def test_trained_readers_learn_separable_hands(rng: np.random.Generator) -> None:
    # Early stopping needs several batches per epoch, so this cannot be much smaller
    points, targets = labelled_hands(rng, 2000)
    inputs = features(points)
    network = train_network(inputs, targets, (16,), 1e-4, 200, seed=0)
    logistic = train_logistic(inputs, targets)
    test_points, test_targets = labelled_hands(rng, 100)
    for reader in (network, logistic):
        accuracy = np.mean(
            reader.probabilities(features(test_points)).argmax(axis=1) == test_targets
        )
        assert accuracy > 0.9


def test_training_needs_every_class(rng: np.random.Generator) -> None:
    points, targets = labelled_hands(rng, 40)
    keep = targets < 3
    with pytest.raises(ValueError):
        train_logistic(features(points[keep]), targets[keep])


def test_network_round_trips_through_json(rng: np.random.Generator) -> None:
    network = GestureNetwork(
        weights=(rng.normal(size=(FEATURE_COUNT, 5)), rng.normal(size=(5, 4))),
        biases=(rng.normal(size=5), rng.normal(size=4)),
        threshold=0.73,
    )
    restored = GestureNetwork.from_json(network.to_json())
    inputs = rng.normal(size=(6, FEATURE_COUNT))
    assert restored.probabilities(inputs) == pytest.approx(network.probabilities(inputs), rel=1e-4)
    assert restored.probabilities(inputs).sum(axis=1) == pytest.approx(np.ones(6))
    assert restored.threshold == 0.73
    assert network.rounded().weights[0][0, 0] == float(f'{network.weights[0][0, 0]:.6g}')
    assert network.with_threshold(0.5).threshold == 0.5


@pytest.mark.parametrize('change', [{'format': 2}, {'classes': ['a', 'b', 'c', 'd']}])
def test_network_rejects_other_exports(change: dict[str, object]) -> None:
    payload = GestureNetwork(
        weights=(np.zeros((FEATURE_COUNT, 4)),), biases=(np.zeros(4),)
    ).to_json()
    with pytest.raises(ValueError):
        GestureNetwork.from_json({**payload, **change})


def test_network_checks_its_shapes() -> None:
    with pytest.raises(ValueError):
        GestureNetwork(weights=(np.zeros((3, 4)),), biases=(np.zeros(4),))
    with pytest.raises(ValueError):
        GestureNetwork(weights=(np.zeros((FEATURE_COUNT, 3)),), biases=(np.zeros(3),))
    with pytest.raises(ValueError):
        GestureNetwork(weights=(), biases=())


def test_decide_is_unsure_below_the_threshold() -> None:
    probabilities = np.array([[0.5, 0.3, 0.1, 0.1], [0.05, 0.9, 0.03, 0.02]])
    assert decide(probabilities, 0.8).tolist() == [UNSURE, PAPER]


def test_wrong_move_rate_counts_misread_moves_and_played_non_moves() -> None:
    targets = np.array([ROCK, PAPER, OTHER, OTHER])
    decisions = np.array([PAPER, UNSURE, SCISSORS, OTHER])
    assert wrong_move_rate(decisions, targets) == 0.5


def test_choose_threshold_finds_the_lowest_safe_value() -> None:
    probabilities = np.array([[0.6, 0.4, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0]])
    targets = np.array([PAPER, ROCK])
    assert choose_threshold(probabilities, targets, max_rate=0.0) == 0.61
    assert choose_threshold(one_hot(np.array([ROCK])), np.array([PAPER]), 0.0) == 0.99


def test_evaluate_reports_game_metrics() -> None:
    probabilities = one_hot(np.array([ROCK, PAPER, SCISSORS, OTHER, ROCK]))
    targets = np.array([ROCK, PAPER, ROCK, OTHER, OTHER])
    metrics = evaluate(probabilities, targets, threshold=0.5)
    assert metrics.accuracy == 0.6
    assert metrics.moves_read_correctly == pytest.approx(2 / 3)
    assert metrics.others_played_as_moves == 0.5
    assert metrics.wrong_move_rate == 0.4
    assert metrics.confusion[2][0] == 0


def test_annotation_files_skip_two_handed_gestures_and_notebook_copies() -> None:
    names = [
        'annotations/',
        'annotations/train/fist.json',
        'annotations/train/hand_heart.json',
        'annotations/test/.ipynb_checkpoints/fist-checkpoint.json',
        'annotations/other/fist.json',
    ]
    assert annotation_files(names) == [('train', 'annotations/train/fist.json')]


def test_extract_landmarks_keeps_complete_hands(tmp_path: Path) -> None:
    archive = hagrid_archive(tmp_path / 'annotations.zip', hands_per_label=10)
    table = extract_landmarks(archive)
    # Ten hands per gesture and split, minus one without landmarks, plus two relaxed hands
    assert len(table) == 3 * len(FINGERS) * (10 - 1 + 2)
    assert table.points.shape[1:] == (21, 2)
    assert set(table.label_names) == {*FINGERS, 'no_gesture'}
    path = tmp_path / 'table.npz'
    table.save(path)
    restored = LandmarkTable.load(path)
    assert restored.label_names == table.label_names
    assert np.array_equal(restored.points, table.points)


def test_draw_sample_balances_classes(tmp_path: Path) -> None:
    table = extract_landmarks(hagrid_archive(tmp_path / 'annotations.zip', hands_per_label=20))
    sample = draw_sample(table, 'train', CLASS_OF, hands_per_class=12, seed=0)
    assert np.bincount(sample.targets).tolist() == [12, 12, 12, 12]
    assert 'three' not in sample.labels
    others = sample.labels[sample.targets == OTHER]
    assert set(others) == {'one', 'no_gesture'}
    assert len(sample.users) == len(sample)


def test_spread_takes_evenly_and_fills_from_larger_pools() -> None:
    pools = [np.arange(2), np.arange(10, 20), np.arange(30, 40)]
    taken = _spread(pools, 12)
    assert [len(part) for part in taken] == [2, 5, 5]
    assert [len(part) for part in _spread([np.arange(3)], 10)] == [3]
