'''Replaying recorded human games against AI strategies.

The recorded players faced a different computer program. In a replay the AI predicts each move
from the history the player actually saw, and its own choice only decides the score. This
measures how well the AI predicts real people, but not how people would react to it.
'''

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from rpsai.humans import HumanGames
from rpsai.moves import counter, outcome
from rpsai.predictors import (
    TIE_TOLERANCE,
    Distribution,
    MovePredictor,
    PredictorSettings,
    choose_move,
)


class Opponent(Protocol):
    '''The AI side of a game.'''

    def move(self, draw: float) -> tuple[int, Distribution | None]:
        '''Choose a move, returning the predicted distribution of the player's move if any.'''

    def observe(self, player: int, opponent: int) -> None:
        '''Learn from the round the player saw: their move and the move they faced.'''


class RandomOpponent:
    '''Plays each move with probability one third, which wins a third of rounds against anyone.'''

    def move(self, draw: float) -> tuple[int, Distribution | None]:
        '''Pick at random.'''
        return min(int(draw * 3), 2), None

    def observe(self, player: int, opponent: int) -> None:
        '''Ignore the round.'''


class BeatLastOpponent:
    '''Plays what beats the player's previous move, a common rule of thumb.'''

    def __init__(self) -> None:
        self._last: int | None = None

    def move(self, draw: float) -> tuple[int, Distribution | None]:
        '''Counter the last move, or pick at random in the first round.'''
        if self._last is None:
            return min(int(draw * 3), 2), None
        return counter(self._last), None

    def observe(self, player: int, opponent: int) -> None:
        '''Remember the player's move.'''
        self._last = player


class PredictorOpponent:
    '''Plays the move with the best expected score against the predictor's forecast.'''

    def __init__(self, settings: PredictorSettings, experts: Sequence[str] | None = None) -> None:
        self.predictor = MovePredictor(settings, experts)

    def move(self, draw: float) -> tuple[int, Distribution | None]:
        '''Forecast the player's move and respond to it.'''
        forecast = self.predictor.forecast()
        return choose_move(forecast.mixture, self.predictor.settings.margin, draw), forecast.mixture

    def observe(self, player: int, opponent: int) -> None:
        '''Update the experts and their weights.'''
        self.predictor.observe(player, opponent)


OpponentFactory = Callable[[], Opponent]


def prediction_credit(forecast: Distribution, move: int) -> float:
    '''Score a forecast 1 when move was its most likely, sharing the credit among ties.'''
    best = max(forecast)
    tied = [candidate for candidate in range(3) if forecast[candidate] >= best - TIE_TOLERANCE]
    return 1 / len(tied) if move in tied else 0.0


@dataclass(frozen=True)
class Replay:
    '''Round-by-round results of one AI strategy against every recorded player.'''

    # AI result per player and round: 1 win, 0 draw, -1 loss
    results: np.ndarray
    # 1 when the player chose the move the AI thought most likely, for predicting strategies.
    # Moves tied for most likely share the credit, so a forecast with no opinion scores a third.
    correct: np.ndarray | None
    # Negative log probability the AI gave the player's actual move
    log_loss: np.ndarray | None


def replay_games(games: HumanGames, make_opponent: OpponentFactory, seed: int) -> Replay:
    '''Play every recorded game against a fresh AI. The same seed gives the same random draws.'''
    draws = np.random.default_rng(seed).random(games.player.shape)
    players, rounds = games.player.shape
    results = np.zeros((players, rounds), dtype=np.int64)
    correct = np.zeros((players, rounds))
    log_loss = np.zeros((players, rounds))
    predicts = True
    for row in range(players):
        opponent = make_opponent()
        for column in range(rounds):
            player_move = int(games.player[row, column])
            ai_move, forecast = opponent.move(float(draws[row, column]))
            results[row, column] = outcome(ai_move, player_move)
            if forecast is None:
                predicts = False
            else:
                correct[row, column] = prediction_credit(forecast, player_move)
                log_loss[row, column] = -math.log(forecast[player_move])
            opponent.observe(player_move, int(games.opponent[row, column]))
    return Replay(
        results=results,
        correct=correct if predicts else None,
        log_loss=log_loss if predicts else None,
    )


@dataclass(frozen=True)
class Interval:
    '''A point estimate with a 95% bootstrap interval.'''

    value: float
    low: float
    high: float


@dataclass(frozen=True)
class StrategySummary:
    '''Averages over players, with the spread between players as bootstrap intervals.'''

    name: str
    win_rate: Interval
    draw_rate: float
    loss_rate: float
    net_score: Interval
    accuracy: float | None
    log_loss: float | None
    # Win rate in each block of ten rounds
    win_rate_by_block: tuple[float, ...]


def bootstrap(per_player: np.ndarray, samples: int, seed: int) -> Interval:
    '''Mean over players with a percentile interval from resampling players.'''
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(per_player), size=(samples, len(per_player)))
    means = per_player[rows].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Interval(float(per_player.mean()), float(low), float(high))


def summarise(
    name: str, replay: Replay, samples: int, seed: int, block: int = 10
) -> StrategySummary:
    '''Summarise a replay. Intervals resample players, because rounds of one player are related.'''
    wins = (replay.results == 1).mean(axis=1)
    losses = (replay.results == -1).mean(axis=1)
    rounds = replay.results.shape[1]
    blocks = tuple(
        float((replay.results[:, start : start + block] == 1).mean())
        for start in range(0, rounds, block)
    )
    return StrategySummary(
        name=name,
        win_rate=bootstrap(wins, samples, seed),
        draw_rate=float((replay.results == 0).mean()),
        loss_rate=float(losses.mean()),
        net_score=bootstrap(wins - losses, samples, seed),
        accuracy=None if replay.correct is None else float(replay.correct.mean()),
        log_loss=None if replay.log_loss is None else float(replay.log_loss.mean()),
        win_rate_by_block=blocks,
    )


def _mean_log_loss(games: HumanGames, settings: PredictorSettings, seed: int) -> float:
    replay = replay_games(games, lambda: PredictorOpponent(settings), seed)
    if replay.log_loss is None:
        raise RuntimeError('the predictor returned no forecast')
    return float(replay.log_loss.mean())


def _mean_net_score(games: HumanGames, settings: PredictorSettings, seed: int) -> float:
    results = replay_games(games, lambda: PredictorOpponent(settings), seed).results
    return float((results == 1).mean() - (results == -1).mean())


@dataclass(frozen=True)
class TuningResult:
    '''The chosen settings and the score of every candidate.'''

    settings: PredictorSettings
    candidates: tuple[tuple[PredictorSettings, float], ...]
    margins: tuple[tuple[float, float], ...]


def tune(
    games: HumanGames,
    base: PredictorSettings,
    grid: dict[str, Sequence[float]],
    margins: Sequence[float],
    seed: int,
) -> TuningResult:
    '''Pick the settings with the lowest log loss, then the margin with the best net score.

    Log loss rewards honest probabilities and is far less noisy than wins, which depend on a
    third of a point per round. The margin only changes which move is played, so it is chosen
    on net score afterwards.
    '''
    names = list(grid)
    combinations: list[dict[str, Any]] = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]
    candidates = [replace(base, **combination) for combination in combinations]
    losses = [_mean_log_loss(games, candidate, seed) for candidate in candidates]
    best = candidates[int(np.argmin(losses))]
    with_margins = [replace(best, margin=margin) for margin in margins]
    nets = [_mean_net_score(games, candidate, seed) for candidate in with_margins]
    chosen_margin = margins[int(np.argmax(nets))]
    return TuningResult(
        settings=replace(best, margin=chosen_margin),
        candidates=tuple(zip(candidates, losses, strict=True)),
        margins=tuple(zip(margins, nets, strict=True)),
    )
