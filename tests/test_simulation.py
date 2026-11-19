from pathlib import Path

import numpy as np
import pytest
from synthetic import human_games_archive

from rpsai.humans import (
    PLAYERS,
    SOURCE_ENCODING,
    HumanGames,
    load_human_games,
    parse_series,
    split_players,
)
from rpsai.moves import Move
from rpsai.predictor_pipeline import describe_habits, mean_weights
from rpsai.predictors import EXPERT_NAMES, UNIFORM, PredictorSettings
from rpsai.simulation import (
    BeatLastOpponent,
    PredictorOpponent,
    RandomOpponent,
    bootstrap,
    prediction_credit,
    replay_games,
    summarise,
    tune,
)


def games_of(player: list[list[int]], opponent: list[list[int]] | None = None) -> HumanGames:
    moves = np.array(player)
    against = np.zeros_like(moves) if opponent is None else np.array(opponent)
    return HumanGames(np.arange(1, len(moves) + 1), moves, against)


def test_parse_series_maps_the_source_encoding_and_ignores_the_trailing_comma() -> None:
    assert parse_series('0,1,2,') == [Move.ROCK, Move.SCISSORS, Move.PAPER]
    assert set(SOURCE_ENCODING.values()) == set(Move)


def test_load_human_games_skips_players_whose_files_do_not_line_up(tmp_path: Path) -> None:
    games = load_human_games(human_games_archive(tmp_path / 'games.zip', rounds=12))
    assert len(games) == PLAYERS - 1
    assert 7 not in games.player_ids
    assert games.player.shape == (PLAYERS - 1, 12)


def test_human_games_validate_their_shapes() -> None:
    with pytest.raises(ValueError):
        HumanGames(np.arange(2), np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        HumanGames(np.arange(3), np.zeros((2, 3)), np.zeros((2, 3)))


def test_split_players_is_disjoint_and_complete() -> None:
    games = games_of([[0, 1]] * 10)
    tuning, test = split_players(games, 0.3, seed=0)
    assert len(tuning) == 3
    assert len(test) == 7
    assert set(tuning.player_ids) | set(test.player_ids) == set(games.player_ids)
    assert not set(tuning.player_ids) & set(test.player_ids)


def test_prediction_credit_shares_ties() -> None:
    assert prediction_credit((0.2, 0.5, 0.3), 1) == 1.0
    assert prediction_credit((0.2, 0.5, 0.3), 0) == 0.0
    assert prediction_credit(UNIFORM, 2) == pytest.approx(1 / 3)


def test_random_opponent_wins_about_a_third(rng: np.random.Generator) -> None:
    games = games_of(rng.integers(0, 3, (200, 30)).tolist())
    replay = replay_games(games, RandomOpponent, seed=1)
    assert replay.correct is None
    assert replay.log_loss is None
    assert (replay.results == 1).mean() == pytest.approx(1 / 3, abs=0.03)


def test_beat_last_opponent_beats_a_player_who_repeats() -> None:
    replay = replay_games(games_of([[2] * 10]), BeatLastOpponent, seed=0)
    assert (replay.results[0, 1:] == 1).all()


def test_predictor_opponent_beats_a_predictable_player(settings: PredictorSettings) -> None:
    replay = replay_games(
        games_of([[round_number % 3 for round_number in range(40)]] * 3),
        lambda: PredictorOpponent(settings),
        seed=0,
    )
    assert replay.correct is not None
    assert replay.log_loss is not None
    assert (replay.results[:, 10:] == 1).mean() > 0.9
    assert replay.log_loss[:, 0] == pytest.approx(np.log(3))


def test_bootstrap_brackets_the_mean() -> None:
    values = np.array([0.2, 0.4, 0.6, 0.8])
    interval = bootstrap(values, samples=500, seed=0)
    assert interval.low <= interval.value <= interval.high
    assert interval.value == pytest.approx(0.5)


def test_summarise_reports_blocks_of_rounds(settings: PredictorSettings) -> None:
    replay = replay_games(games_of([[0] * 25] * 4), lambda: PredictorOpponent(settings), seed=0)
    summary = summarise('test', replay, samples=100, seed=0, block=10)
    assert len(summary.win_rate_by_block) == 3
    assert summary.win_rate.value + summary.draw_rate + summary.loss_rate == pytest.approx(1.0)
    assert summary.accuracy is not None


def test_tune_tries_every_combination(settings: PredictorSettings) -> None:
    games = games_of([[round_number % 3 for round_number in range(20)]] * 4)
    result = tune(
        games,
        settings,
        {'decay': [0.5, 1.0], 'learning_rate': [1.0, 3.0]},
        margins=[0.0, 0.2],
        seed=0,
    )
    assert len(result.candidates) == 4
    assert len(result.margins) == 2
    assert result.settings.decay in (0.5, 1.0)
    assert result.settings.margin in (0.0, 0.2)


def test_describe_habits_counts_repeats_after_each_result() -> None:
    # Rock against scissors is a win, then the player repeats rock
    games = games_of([[Move.ROCK, Move.ROCK, Move.PAPER]], [[Move.SCISSORS, Move.ROCK, Move.ROCK]])
    habits = describe_habits(games)
    assert habits.repeat_after[2] == 1.0
    assert habits.repeat_after[1] == 0.0
    assert np.isnan(habits.repeat_after[0])
    assert habits.repeat_rate == 0.5
    assert sum(habits.move_shares) == pytest.approx(1.0)


def test_mean_weights_has_one_row_per_round(settings: PredictorSettings) -> None:
    weights = mean_weights(games_of([[0, 1, 2, 0]] * 2), settings)
    assert weights.shape == (4, len(EXPERT_NAMES))
    assert weights.sum(axis=1) == pytest.approx(np.ones(4))
