'''Files for the web game: the models, and fixtures that keep Python and JavaScript in step.'''

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rpsai.gestures import GestureNetwork, decide
from rpsai.landmarks import features
from rpsai.moves import counter
from rpsai.predictors import EXPERT_NAMES, MovePredictor, PredictorSettings, choose_move

GESTURE_MODEL_FILE = 'gesture_reader.json'
PREDICTOR_FILE = 'predictor.json'
GESTURE_FIXTURE_FILE = 'gesture_cases.json'
PREDICTOR_FIXTURE_FILE = 'predictor_cases.json'
EXPORT_FORMAT = 1
FIXTURE_SEED = 7
FIXTURE_HANDS = 24
FIXTURE_ROUNDS = 40

ATTRIBUTION = (
    'A classifier trained on the hand landmarks of the HaGRID v2 annotations by Nuzhdin, Nagaev, '
    'Sautin, Kapitanov, and Kvanchiani (https://github.com/hukenovs/hagrid), which are licensed '
    'under a variant of CC BY-SA 4.0. These weights are licensed under CC BY-SA 4.0: '
    'https://creativecommons.org/licenses/by-sa/4.0/'
)

History = list[tuple[int, int]]
ScriptedPlayer = Callable[[History, float], int]


def write_json(payload: Any, path: Path, indent: int | None = None) -> Path:
    '''Write JSON with a trailing newline and Unix line endings.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = (',', ':') if indent is None else None
    text = json.dumps(payload, indent=indent, separators=separators)
    path.write_text(text + '\n', encoding='utf-8', newline='\n')
    return path


def gesture_model_payload(network: GestureNetwork) -> dict[str, Any]:
    '''The gesture reader as the game loads it.'''
    return {**network.to_json(), 'attribution': ATTRIBUTION}


def predictor_payload(settings: PredictorSettings) -> dict[str, Any]:
    '''The tuned predictor settings as the game loads them.'''
    return {'format': EXPORT_FORMAT, 'experts': list(EXPERT_NAMES), 'settings': asdict(settings)}


def gesture_fixture(network: GestureNetwork, seed: int = FIXTURE_SEED) -> dict[str, Any]:
    '''Random landmark sets with the features, probabilities, and decision Python computes.'''
    rng = np.random.default_rng(seed)
    points = rng.random((FIXTURE_HANDS, 21, 2))
    aspects = rng.uniform(0.5, 2.0, FIXTURE_HANDS)
    inputs = features(points, aspects)
    probabilities = network.probabilities(inputs)
    decisions = decide(probabilities, network.threshold)
    return {
        'threshold': network.threshold,
        'cases': [
            {
                'points': points[index].tolist(),
                'aspect': float(aspects[index]),
                'features': inputs[index].tolist(),
                'probabilities': probabilities[index].tolist(),
                'decision': int(decisions[index]),
            }
            for index in range(FIXTURE_HANDS)
        ],
    }


def _scripted_players(rng: np.random.Generator) -> dict[str, ScriptedPlayer]:
    def always_rock(history: History, draw: float) -> int:
        return 0

    def cycles(history: History, draw: float) -> int:
        return (history[-1][0] + 1) % 3 if history else 0

    def win_stay_lose_shift(history: History, draw: float) -> int:
        if not history:
            return 1
        player, ai = history[-1]
        return player if player == counter(ai) else counter(player)

    def random_player(history: History, draw: float) -> int:
        return int(rng.integers(3))

    return {
        'always rock': always_rock,
        'cycles': cycles,
        'win-stay lose-shift': win_stay_lose_shift,
        'random': random_player,
    }


def predictor_fixture(settings: PredictorSettings, seed: int = FIXTURE_SEED) -> dict[str, Any]:
    '''Games against scripted players with every forecast and move, for two sets of settings.'''
    variants = [settings, replace(settings, margin=0.05, pattern_length=3, decay=0.8)]
    cases = []
    for variant in variants:
        rng = np.random.default_rng(seed)
        games = []
        for name, player in _scripted_players(rng).items():
            predictor = MovePredictor(variant)
            history: History = []
            rounds = []
            for draw in rng.random(FIXTURE_ROUNDS):
                forecast = predictor.forecast()
                ai = choose_move(forecast.mixture, variant.margin, float(draw))
                move = player(history, float(draw))
                rounds.append(
                    {
                        'draw': float(draw),
                        'mixture': list(forecast.mixture),
                        'weights': list(forecast.weights),
                        'ai': ai,
                        'player': move,
                    }
                )
                predictor.observe(move, ai)
                history.append((move, ai))
            games.append({'name': name, 'rounds': rounds})
        cases.append({'settings': asdict(variant), 'games': games})
    return {'experts': list(EXPERT_NAMES), 'cases': cases}
