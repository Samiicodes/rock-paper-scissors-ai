'''Synthetic hands and archives laid out like the real downloads, so tests need no data.'''

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from rpsai.humans import PLAYERS, SERIES_DIR

# Which of index, middle, ring, and little finger are straight for each synthetic gesture
FINGERS = {
    'fist': (False, False, False, False),
    'palm': (True, True, True, True),
    'peace': (True, True, False, False),
    'one': (True, False, False, False),
    'three': (True, True, True, False),
}
# Finger bases in hand coordinates: wrist at the origin, middle finger base at (0, -1)
_BASES = {5: (0.3, -0.95), 9: (0.0, -1.0), 13: (-0.3, -0.92), 17: (-0.55, -0.8)}
_THUMB = ((0.35, -0.3), (0.55, -0.5), (0.7, -0.65), (0.8, -0.8))
_STRAIGHT = (-0.45, -0.75, -1.0)
_CURLED = (-0.3, -0.1, 0.1)


def hand_in_hand_coordinates(fingers: tuple[bool, bool, bool, bool]) -> np.ndarray:
    '''An upright hand with the given fingers straight, before any placement.'''
    points = np.zeros((21, 2))
    points[1:5] = _THUMB
    for straight, (base, (x, y)) in zip(fingers, _BASES.items(), strict=True):
        points[base] = (x, y)
        for joint, offset in enumerate(_STRAIGHT if straight else _CURLED, start=1):
            points[base + joint] = (x, y + offset)
    return points


def make_hand(
    fingers: tuple[bool, bool, bool, bool], rng: np.random.Generator, noise: float = 0.02
) -> np.ndarray:
    '''Place a hand in an image at a random angle, size, and position, mirrored half the time.'''
    points = hand_in_hand_coordinates(fingers) + rng.normal(0, noise, (21, 2))
    if rng.random() < 0.5:
        points[:, 0] = -points[:, 0]
    angle = rng.uniform(0, 2 * np.pi)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = rng.uniform(0.06, 0.12)
    centre = rng.uniform(0.35, 0.65, size=2)
    return np.asarray(points @ rotation.T * scale + centre)


def hagrid_archive(path: Path, hands_per_label: int = 40, seed: int = 0) -> Path:
    '''Write annotations in HaGRID's layout: one JSON file per split and gesture.'''
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path, 'w') as archive:
        for split_index, split in enumerate(('train', 'val', 'test')):
            for label, fingers in FINGERS.items():
                images = {}
                for index in range(hands_per_label):
                    user = f'{split}-user-{index % 7}'
                    hands = [make_hand(fingers, rng).tolist()]
                    labels = [label]
                    if index % 5 == 0:
                        a, b, c, d = (bool(value) for value in rng.integers(0, 2, 4))
                        hands.append(make_hand((a, b, c, d), rng).tolist())
                        labels.append('no_gesture')
                    if index == 0:
                        hands[0] = []
                    images[f'{split}-{label}-{index}'] = {
                        'bboxes': [[0.1, 0.1, 0.2, 0.2]] * len(hands),
                        'labels': labels,
                        'united_bbox': None,
                        'united_label': None,
                        'user_id': f'{user}-{split_index}',
                        'hand_landmarks': hands,
                        'meta': {'age': [30.0], 'gender': ['female'], 'race': ['White']},
                    }
                archive.writestr(f'annotations/{split}/{label}.json', json.dumps(images))
            archive.writestr(f'annotations/{split}/hand_heart.json', json.dumps({}))
            archive.writestr(f'annotations/{split}/.ipynb_checkpoints/fist-checkpoint.json', '{}')
        archive.writestr('annotations/', '')
    return path


def cycling_player(rounds: int, rng: np.random.Generator) -> list[int]:
    '''A player who usually steps from rock to scissors to paper, in the dataset's encoding.'''
    moves = [int(rng.integers(3))]
    for _ in range(rounds - 1):
        moves.append((moves[-1] + 1) % 3 if rng.random() < 0.8 else int(rng.integers(3)))
    return moves


def human_games_archive(path: Path, rounds: int = 20, seed: int = 0) -> Path:
    '''Write games in the layout of the kuro-lab/RPSdata repository, with one broken player.'''
    rng = np.random.default_rng(seed)
    root = f'RPSdata-0000000/{SERIES_DIR}'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{root}/', '')
        for player in range(1, PLAYERS + 1):
            human = cycling_player(rounds, rng)
            computer = [int(value) for value in rng.integers(0, 3, rounds - (player == 7))]
            archive.writestr(
                f'{root}/Human_hand_data/{player}.csv', ','.join(map(str, human)) + ','
            )
            archive.writestr(
                f'{root}/AI_hand_data/{player}.csv', ','.join(map(str, computer)) + ','
            )
    return path


def zip_bytes(files: dict[str, str]) -> bytes:
    '''An in-memory zip archive with the given text files.'''
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()
