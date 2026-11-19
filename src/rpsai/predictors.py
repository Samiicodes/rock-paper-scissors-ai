'''Online move prediction: experts that each model one habit, combined by how well they predict.

The web game runs a JavaScript copy of this module in web/src/predictor.js. Both are checked against
the same fixtures in web/test/fixtures, so a change here must be made there too.
'''

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from rpsai.moves import outcome

Distribution = tuple[float, float, float]

UNIFORM: Distribution = (1 / 3, 1 / 3, 1 / 3)
# Expected scores within this distance of the best count as a tie, so that Python and JavaScript
# agree even when their exp and log differ in the last bit
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PredictorSettings:
    '''Hyperparameters of the predictor, tuned on held-out human games.'''

    # Each expert's counts for a situation shrink by this factor whenever that situation recurs
    decay: float
    # Pseudo-count added to every move before counts become probabilities
    prior: float
    # How sharply the combiner favours experts with a lower log loss
    learning_rate: float
    # How quickly the combiner forgets an expert's old mistakes
    score_decay: float
    # Longest stretch of recent rounds that the pattern expert looks for earlier in the game
    pattern_length: int
    # The AI plays at random unless the expected score of its best move is above this
    margin: float

    def __post_init__(self) -> None:
        if not 0 < self.decay <= 1:
            raise ValueError('decay must be in (0, 1]')
        if self.prior <= 0:
            raise ValueError('prior must be positive')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if not 0 < self.score_decay <= 1:
            raise ValueError('score_decay must be in (0, 1]')
        if self.pattern_length < 1:
            raise ValueError('pattern_length must be at least 1')
        if self.margin < 0:
            raise ValueError('margin must not be negative')


def normalise(counts: Sequence[float], prior: float) -> Distribution:
    '''Turn move counts into probabilities, adding the prior to each move.'''
    total = counts[0] + counts[1] + counts[2] + 3 * prior
    return (
        (counts[0] + prior) / total,
        (counts[1] + prior) / total,
        (counts[2] + prior) / total,
    )


def _add(row: list[float], index: int, decay: float) -> None:
    for position in range(3):
        row[position] *= decay
    row[index] += 1.0


class Expert(ABC):
    '''Predicts the player's next move from the rounds it has seen so far.'''

    name: str

    @abstractmethod
    def predict(self) -> Distribution:
        '''Return the probability of each move in the next round.'''

    @abstractmethod
    def observe(self, player: int, ai: int) -> None:
        '''Learn from a finished round.'''


class UniformExpert(Expert):
    '''Assumes the player is unpredictable. It wins the combiner's trust when nothing else does.'''

    name = 'uniform'

    def predict(self) -> Distribution:
        '''Return equal probabilities.'''
        return UNIFORM

    def observe(self, player: int, ai: int) -> None:
        '''Ignore the round.'''


class FrequencyExpert(Expert):
    '''Expects the move the player has chosen most often recently.'''

    name = 'frequency'

    def __init__(self, decay: float, prior: float) -> None:
        self._decay = decay
        self._prior = prior
        self._counts = [0.0, 0.0, 0.0]

    def predict(self) -> Distribution:
        '''Return recent move frequencies.'''
        return normalise(self._counts, self._prior)

    def observe(self, player: int, ai: int) -> None:
        '''Count the player's move.'''
        _add(self._counts, player, self._decay)


class MarkovExpert(Expert):
    '''Expects the move that has followed the player's last few moves.'''

    def __init__(self, order: int, decay: float, prior: float) -> None:
        self.name = f'markov{order}'
        self._order = order
        self._decay = decay
        self._prior = prior
        self._history: list[int] = []
        self._table: dict[tuple[int, ...], list[float]] = {}

    def predict(self) -> Distribution:
        '''Return what followed the current context before.'''
        if len(self._history) < self._order:
            return UNIFORM
        row = self._table.get(tuple(self._history[-self._order :]))
        return normalise(row if row is not None else (0.0, 0.0, 0.0), self._prior)

    def observe(self, player: int, ai: int) -> None:
        '''Record the move under the context that preceded it.'''
        if len(self._history) >= self._order:
            context = tuple(self._history[-self._order :])
            _add(self._table.setdefault(context, [0.0, 0.0, 0.0]), player, self._decay)
        self._history.append(player)


class ShiftExpert(Expert):
    '''Expects a step from the player's last move: staying, moving up a move, or moving down.

    With by_result the steps are counted separately after a win, a draw, and a loss, which
    captures habits such as repeating a winning move. Without it the expert captures players who
    cycle through the moves. Stepping relative to the AI's last move instead would add nothing:
    given the result and the player's move, the AI's move is already known.
    '''

    def __init__(self, name: str, by_result: bool, decay: float, prior: float) -> None:
        self.name = name
        self._by_result = by_result
        self._decay = decay
        self._prior = prior
        self._last: tuple[int, int] | None = None
        # One row of step counts for each result of the previous round: loss, draw, win
        self._table = [[0.0, 0.0, 0.0] for _ in range(3)]

    def _row(self, player: int, ai: int) -> list[float]:
        return self._table[outcome(player, ai) + 1 if self._by_result else 0]

    def predict(self) -> Distribution:
        '''Return the steps seen in the same situation, applied to the last move.'''
        if self._last is None:
            return UNIFORM
        player, ai = self._last
        steps = normalise(self._row(player, ai), self._prior)
        return (
            steps[(0 - player) % 3],
            steps[(1 - player) % 3],
            steps[(2 - player) % 3],
        )

    def observe(self, player: int, ai: int) -> None:
        '''Record the step from the previous move.'''
        if self._last is not None:
            last_player, last_ai = self._last
            _add(self._row(last_player, last_ai), (player - last_player) % 3, self._decay)
        self._last = (player, ai)


class PatternExpert(Expert):
    '''Finds the longest run of recent rounds that happened before and expects what followed it.'''

    name = 'pattern'

    def __init__(self, max_length: int, prior: float) -> None:
        self._max_length = max_length
        self._prior = prior
        # Each round is stored as player * 3 + ai so that both moves must match
        self._history: list[int] = []

    def predict(self) -> Distribution:
        '''Return the moves that followed the longest earlier match of the recent rounds.'''
        history = self._history
        size = len(history)
        for length in range(min(self._max_length, size - 1), 0, -1):
            suffix = history[size - length :]
            counts = [0.0, 0.0, 0.0]
            found = False
            for end in range(length, size):
                if history[end - length : end] == suffix:
                    counts[history[end] // 3] += 1.0
                    found = True
            if found:
                return normalise(counts, self._prior)
        return UNIFORM

    def observe(self, player: int, ai: int) -> None:
        '''Append the round.'''
        self._history.append(player * 3 + ai)


EXPERT_NAMES = (
    'uniform',
    'frequency',
    'cycle',
    'after_result',
    'markov1',
    'markov2',
    'pattern',
)


def build_experts(settings: PredictorSettings) -> tuple[Expert, ...]:
    '''Create one of every expert, in the order of EXPERT_NAMES.'''
    return (
        UniformExpert(),
        FrequencyExpert(settings.decay, settings.prior),
        ShiftExpert('cycle', False, settings.decay, settings.prior),
        ShiftExpert('after_result', True, settings.decay, settings.prior),
        MarkovExpert(1, settings.decay, settings.prior),
        MarkovExpert(2, settings.decay, settings.prior),
        PatternExpert(settings.pattern_length, settings.prior),
    )


@dataclass(frozen=True)
class Forecast:
    '''The predictor's view of the next round.'''

    mixture: Distribution
    names: tuple[str, ...]
    predictions: tuple[Distribution, ...]
    weights: tuple[float, ...]


class MovePredictor:
    '''Combines the experts with exponential weights on their recent log loss.'''

    def __init__(self, settings: PredictorSettings, experts: Sequence[str] | None = None) -> None:
        chosen = EXPERT_NAMES if experts is None else tuple(experts)
        unknown = set(chosen) - set(EXPERT_NAMES)
        if unknown or not chosen:
            raise ValueError(f'unknown or missing experts: {sorted(unknown)}')
        self.settings = settings
        self.experts = tuple(e for e in build_experts(settings) if e.name in chosen)
        self.losses = [0.0] * len(self.experts)
        self.rounds = 0

    def forecast(self) -> Forecast:
        '''Predict the player's next move.'''
        predictions = tuple(expert.predict() for expert in self.experts)
        best = min(self.losses)
        raw = [math.exp(-self.settings.learning_rate * (loss - best)) for loss in self.losses]
        total = 0.0
        for value in raw:
            total += value
        weights = tuple(value / total for value in raw)
        mixture = [0.0, 0.0, 0.0]
        for weight, prediction in zip(weights, predictions, strict=True):
            for move in range(3):
                mixture[move] += weight * prediction[move]
        return Forecast(
            mixture=(mixture[0], mixture[1], mixture[2]),
            names=tuple(expert.name for expert in self.experts),
            predictions=predictions,
            weights=weights,
        )

    def observe(self, player: int, ai: int) -> None:
        '''Score every expert on the finished round, then let each learn from it.'''
        decay = self.settings.score_decay
        for index, expert in enumerate(self.experts):
            probability = expert.predict()[player]
            self.losses[index] = decay * self.losses[index] - math.log(probability)
        for expert in self.experts:
            expert.observe(player, ai)
        self.rounds += 1


def expected_scores(distribution: Distribution) -> Distribution:
    '''Expected score of each AI move against the distribution: 1 for a win, -1 for a loss.'''
    return (
        distribution[2] - distribution[1],
        distribution[0] - distribution[2],
        distribution[1] - distribution[0],
    )


def choose_move(distribution: Distribution, margin: float, draw: float) -> int:
    '''Pick the AI's move. draw is a uniform random number in [0, 1) supplied by the caller.'''
    scores = expected_scores(distribution)
    best = max(scores)
    if best <= margin:
        return min(int(draw * 3), 2)
    tied = [move for move in range(3) if scores[move] >= best - TIE_TOLERANCE]
    return tied[min(int(draw * len(tied)), len(tied) - 1)]
