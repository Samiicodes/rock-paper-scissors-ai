'''Tuning the move predictor on recorded human games and testing it on players it never saw.'''

import logging
from dataclasses import dataclass

import numpy as np

from rpsai.config import AppConfig
from rpsai.humans import PLAYERS, HumanGames, load_human_games, split_players
from rpsai.moves import outcome
from rpsai.predictors import EXPERT_NAMES, MovePredictor, PredictorSettings
from rpsai.simulation import (
    BeatLastOpponent,
    OpponentFactory,
    PredictorOpponent,
    RandomOpponent,
    StrategySummary,
    replay_games,
    summarise,
    tune,
)

logger = logging.getLogger(__name__)

EXPERT_LABELS = {
    'uniform': 'No pattern',
    'frequency': 'Favourite move',
    'cycle': 'Cycling',
    'after_result': 'After a win or loss',
    'markov1': 'After the last move',
    'markov2': 'After the last two moves',
    'pattern': 'Repeated sequence',
}


@dataclass(frozen=True)
class Habits:
    '''How the recorded players behaved, under the assumed move encoding.'''

    # Shares of rock, paper, and scissors
    move_shares: tuple[float, float, float]
    repeat_rate: float
    # Share of rounds in which the player repeated their move, after a loss, draw, and win
    repeat_after: tuple[float, float, float]


@dataclass(frozen=True)
class PredictorResults:
    '''Everything the report needs about the move predictor.'''

    players: int
    tuning_players: int
    test_players: int
    skipped_players: tuple[int, ...]
    rounds: int
    settings: PredictorSettings
    best_log_loss: float
    margins: tuple[tuple[float, float], ...]
    strategies: tuple[StrategySummary, ...]
    habits: Habits
    # Win rate of the combined AI if paper and scissors were swapped in the data, which is the
    # other possible reading of the undocumented move encoding
    swapped_encoding_win_rate: float
    expert_names: tuple[str, ...]
    # Mean combiner weight of each expert in each round, over the test players
    weights_by_round: tuple[tuple[float, ...], ...]


def describe_habits(games: HumanGames) -> Habits:
    '''Summarise move choices and how often players repeat a move after each result.'''
    player, opponent = games.player, games.opponent
    shares = np.bincount(player.ravel(), minlength=3) / player.size
    repeated = player[:, 1:] == player[:, :-1]
    results = np.vectorize(outcome)(player[:, :-1], opponent[:, :-1])
    # A result that never happened has no repeat rate
    repeat_after = tuple(
        float(repeated[results == value].mean()) if (results == value).any() else float('nan')
        for value in (-1, 0, 1)
    )
    return Habits(
        move_shares=(float(shares[0]), float(shares[1]), float(shares[2])),
        repeat_rate=float(repeated.mean()),
        repeat_after=(repeat_after[0], repeat_after[1], repeat_after[2]),
    )


def mean_weights(games: HumanGames, settings: PredictorSettings) -> np.ndarray:
    '''Average the combiner's weight on each expert, round by round.'''
    players, rounds = games.player.shape
    weights = np.zeros((players, rounds, len(EXPERT_NAMES)))
    for row in range(players):
        predictor = MovePredictor(settings)
        for column in range(rounds):
            weights[row, column] = predictor.forecast().weights
            predictor.observe(int(games.player[row, column]), int(games.opponent[row, column]))
    return np.asarray(weights.mean(axis=0))


def run_predictor_pipeline(config: AppConfig) -> PredictorResults:
    '''Tune on one half of the players, then compare strategies on the other half.'''
    settings_config = config.predictor
    games = load_human_games(config.data.raw_dir / config.downloads.human_games.file)
    skipped = tuple(sorted(set(range(1, PLAYERS + 1)) - set(games.player_ids.tolist())))
    tuning_games, test_games = split_players(
        games, settings_config.tuning_fraction, settings_config.seed
    )
    logger.info(
        'Loaded %d players: %d for tuning, %d for testing',
        len(games),
        len(tuning_games),
        len(test_games),
    )

    # Placeholders for the tuned settings, which the grid search replaces
    base = PredictorSettings(
        decay=1.0,
        prior=1.0,
        learning_rate=1.0,
        score_decay=1.0,
        pattern_length=settings_config.pattern_length,
        margin=0.0,
    )
    logger.info('Tuning the predictor')
    tuning = tune(
        tuning_games,
        base,
        settings_config.grid.model_dump(),
        settings_config.margins,
        settings_config.seed,
    )
    settings = tuning.settings
    logger.info(
        'Chose decay %g, prior %g, learning rate %g, score decay %g, margin %g',
        settings.decay,
        settings.prior,
        settings.learning_rate,
        settings.score_decay,
        settings.margin,
    )

    strategies: list[tuple[str, OpponentFactory]] = [
        ('Random', RandomOpponent),
        ('Beat the last move', BeatLastOpponent),
    ]
    strategies += [
        (EXPERT_LABELS[name], _single_expert(settings, name))
        for name in EXPERT_NAMES
        if name != 'uniform'
    ]
    strategies.append(('All predictors combined', lambda: PredictorOpponent(settings)))
    # Every strategy sees the same random draws, so differences come from the strategy alone
    summaries = tuple(
        summarise(
            name,
            replay_games(test_games, factory, settings_config.seed),
            settings_config.bootstrap_samples,
            settings_config.seed,
        )
        for name, factory in strategies
    )
    for summary in summaries:
        logger.info('%s: wins %.1f%% of rounds', summary.name, 100 * summary.win_rate.value)
    swapped = HumanGames(
        test_games.player_ids, (-test_games.player) % 3, (-test_games.opponent) % 3
    )
    swapped_results = replay_games(
        swapped, lambda: PredictorOpponent(settings), settings_config.seed
    ).results

    return PredictorResults(
        players=len(games),
        tuning_players=len(tuning_games),
        test_players=len(test_games),
        skipped_players=skipped,
        rounds=games.player.shape[1],
        settings=settings,
        best_log_loss=min(loss for _, loss in tuning.candidates),
        margins=tuning.margins,
        strategies=summaries,
        habits=describe_habits(games),
        swapped_encoding_win_rate=float((swapped_results == 1).mean()),
        expert_names=EXPERT_NAMES,
        weights_by_round=tuple(
            tuple(float(value) for value in row) for row in mean_weights(test_games, settings)
        ),
    )


def _single_expert(settings: PredictorSettings, name: str) -> OpponentFactory:
    return lambda: PredictorOpponent(settings, [name])
