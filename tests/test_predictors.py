import math
from dataclasses import replace
from typing import Any

import pytest

from rpsai.moves import MOVE_NAMES, Move, counter, outcome
from rpsai.predictors import (
    EXPERT_NAMES,
    UNIFORM,
    FrequencyExpert,
    MarkovExpert,
    MovePredictor,
    PatternExpert,
    PredictorSettings,
    ShiftExpert,
    UniformExpert,
    build_experts,
    choose_move,
    expected_scores,
    normalise,
)

ROCK, PAPER, SCISSORS = Move.ROCK, Move.PAPER, Move.SCISSORS


def test_each_move_beats_the_one_before() -> None:
    assert counter(ROCK) == PAPER
    assert counter(PAPER) == SCISSORS
    assert counter(SCISSORS) == ROCK
    assert MOVE_NAMES == ('rock', 'paper', 'scissors')


@pytest.mark.parametrize(
    ('player', 'opponent', 'expected'),
    [(PAPER, ROCK, 1), (ROCK, PAPER, -1), (SCISSORS, SCISSORS, 0), (ROCK, SCISSORS, 1)],
)
def test_outcome(player: int, opponent: int, expected: int) -> None:
    assert outcome(player, opponent) == expected


@pytest.mark.parametrize(
    'change',
    [
        {'decay': 0.0},
        {'decay': 1.5},
        {'prior': 0.0},
        {'learning_rate': 0.0},
        {'score_decay': 0.0},
        {'pattern_length': 0},
        {'margin': -0.1},
    ],
)
def test_settings_reject_out_of_range_values(
    settings: PredictorSettings, change: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        replace(settings, **change)


def test_normalise_adds_the_prior() -> None:
    assert normalise([2.0, 0.0, 0.0], 1.0) == pytest.approx((0.6, 0.2, 0.2))


def test_uniform_expert_never_learns() -> None:
    expert = UniformExpert()
    expert.observe(ROCK, PAPER)
    assert expert.predict() == UNIFORM


def test_frequency_expert_forgets_old_moves() -> None:
    expert = FrequencyExpert(decay=0.5, prior=0.1)
    for _ in range(5):
        expert.observe(ROCK, PAPER)
    for _ in range(3):
        expert.observe(SCISSORS, PAPER)
    prediction = expert.predict()
    assert prediction[SCISSORS] > prediction[ROCK] > prediction[PAPER]


def test_markov_expert_uses_its_context() -> None:
    expert = MarkovExpert(2, decay=1.0, prior=0.1)
    assert expert.predict() == UNIFORM
    for move in [ROCK, PAPER, SCISSORS] * 3 + [ROCK, PAPER]:
        expert.observe(move, ROCK)
    assert expert.name == 'markov2'
    assert max(range(3), key=expert.predict().__getitem__) == SCISSORS


def test_markov_expert_without_a_matching_context_is_uniform() -> None:
    expert = MarkovExpert(1, decay=1.0, prior=1.0)
    expert.observe(ROCK, ROCK)
    assert expert.predict() == pytest.approx(UNIFORM)


def test_cycle_expert_learns_a_step_from_any_move() -> None:
    expert = ShiftExpert('cycle', by_result=False, decay=1.0, prior=0.1)
    for move in [ROCK, PAPER, SCISSORS, ROCK]:
        expert.observe(move, ROCK)
    # Steps were always up by one, so after rock it expects paper
    assert max(range(3), key=expert.predict().__getitem__) == PAPER


def test_after_result_expert_separates_wins_from_losses() -> None:
    expert = ShiftExpert('after_result', by_result=True, decay=1.0, prior=0.1)
    # The player repeats after a win and switches up after a loss
    rounds = [(ROCK, SCISSORS), (ROCK, PAPER), (PAPER, ROCK), (PAPER, SCISSORS), (SCISSORS, PAPER)]
    for player, ai in rounds * 3:
        expert.observe(player, ai)
    expert.observe(ROCK, SCISSORS)
    assert max(range(3), key=expert.predict().__getitem__) == ROCK
    expert.observe(ROCK, PAPER)
    assert max(range(3), key=expert.predict().__getitem__) == PAPER


def test_pattern_expert_follows_the_longest_earlier_match() -> None:
    expert = PatternExpert(max_length=4, prior=0.01)
    assert expert.predict() == UNIFORM
    for player, ai in [
        (ROCK, ROCK),
        (PAPER, PAPER),
        (SCISSORS, ROCK),
        (ROCK, ROCK),
        (PAPER, PAPER),
    ]:
        expert.observe(player, ai)
    assert max(range(3), key=expert.predict().__getitem__) == SCISSORS


def test_pattern_expert_without_a_match_is_uniform() -> None:
    expert = PatternExpert(max_length=3, prior=1.0)
    expert.observe(ROCK, ROCK)
    expert.observe(PAPER, PAPER)
    assert expert.predict() == UNIFORM


def test_build_experts_follows_the_published_order(settings: PredictorSettings) -> None:
    assert tuple(expert.name for expert in build_experts(settings)) == EXPERT_NAMES


def test_predictor_starts_without_an_opinion(settings: PredictorSettings) -> None:
    forecast = MovePredictor(settings).forecast()
    assert forecast.mixture == pytest.approx(UNIFORM)
    assert sum(forecast.weights) == pytest.approx(1.0)
    assert forecast.names == EXPERT_NAMES


def test_predictor_trusts_the_expert_that_explains_the_player(settings: PredictorSettings) -> None:
    predictor = MovePredictor(settings)
    for round_number in range(30):
        predictor.observe(round_number % 3, ROCK)
    forecast = predictor.forecast()
    weights = dict(zip(forecast.names, forecast.weights, strict=True))
    assert weights['cycle'] > weights['frequency']
    assert weights['cycle'] > weights['uniform']
    assert max(range(3), key=forecast.mixture.__getitem__) == ROCK
    assert predictor.rounds == 30


def test_predictor_can_use_a_subset_of_experts(settings: PredictorSettings) -> None:
    predictor = MovePredictor(settings, ['frequency'])
    assert [expert.name for expert in predictor.experts] == ['frequency']


@pytest.mark.parametrize('experts', [[], ['psychic']])
def test_predictor_rejects_unknown_experts(settings: PredictorSettings, experts: list[str]) -> None:
    with pytest.raises(ValueError):
        MovePredictor(settings, experts)


def test_expected_scores_reward_the_counter_move() -> None:
    assert expected_scores((1.0, 0.0, 0.0)) == (0.0, 1.0, -1.0)


def test_choose_move_plays_the_counter_to_a_clear_favourite() -> None:
    assert choose_move((0.1, 0.2, 0.7), margin=0.0, draw=0.9) == ROCK


def test_choose_move_plays_at_random_without_a_clear_edge() -> None:
    assert [choose_move(UNIFORM, 0.0, draw) for draw in (0.0, 0.5, 0.99)] == [0, 1, 2]
    assert choose_move((0.34, 0.33, 0.33), margin=0.05, draw=0.99) == SCISSORS


def test_choose_move_breaks_ties_with_the_draw() -> None:
    # Rock and paper both score 0.2 against this forecast
    tied = (0.5, 0.1, 0.3)
    scores = expected_scores(tied)
    assert math.isclose(scores[0], scores[1])
    assert {choose_move(tied, 0.0, draw) for draw in (0.1, 0.9)} == {0, 1}
