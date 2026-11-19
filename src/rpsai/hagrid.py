'''Hand landmarks from the HaGRID annotations.

HaGRID (Kapitanov et al.) labels over a million photos of hand gestures, and its annotations
include the 21 MediaPipe hand landmarks of every hand. Only the annotations are needed here, so
no images are downloaded. The annotations are licensed under a variant of CC BY-SA 4.0.
'''

import json
import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpsai.config import SPLITS

logger = logging.getLogger(__name__)

LANDMARKS = 21
# Gestures made with both hands. Their landmark lists are often empty, and the game needs one hand.
TWO_HANDED = frozenset(
    {'hand_heart', 'hand_heart2', 'holy', 'take_picture', 'thumb_index2', 'timeout', 'xsign'}
)


@dataclass(frozen=True)
class LandmarkTable:
    '''Every annotated hand with landmarks: points, gesture label, person, and split.'''

    points: np.ndarray
    label_ids: np.ndarray
    user_ids: np.ndarray
    split_ids: np.ndarray
    label_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.points)

    def save(self, path: Path) -> None:
        '''Store the table as a compressed NumPy archive.'''
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points,
            label_ids=self.label_ids,
            user_ids=self.user_ids,
            split_ids=self.split_ids,
            label_names=np.array(self.label_names),
        )

    @classmethod
    def load(cls, path: Path) -> 'LandmarkTable':
        '''Read a table written by save.'''
        with np.load(path) as archive:
            return cls(
                points=archive['points'],
                label_ids=archive['label_ids'],
                user_ids=archive['user_ids'],
                split_ids=archive['split_ids'],
                label_names=tuple(str(name) for name in archive['label_names']),
            )


def annotation_files(names: list[str]) -> list[tuple[str, str]]:
    '''Return (split, path) for the one-handed gesture files, skipping notebook leftovers.'''
    files = []
    for name in sorted(names):
        parts = name.split('/')
        if len(parts) != 3 or not parts[2].endswith('.json') or parts[1] not in SPLITS:
            continue
        if parts[2].removesuffix('.json') in TWO_HANDED:
            continue
        files.append((parts[1], name))
    return files


def extract_landmarks(archive_path: Path) -> LandmarkTable:
    '''Read every one-handed hand with a full set of landmarks from the annotation archive.'''
    points: list[list[list[float]]] = []
    labels: list[int] = []
    users: list[int] = []
    splits: list[int] = []
    label_index: dict[str, int] = {}
    user_index: dict[str, int] = {}
    skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        for split, name in annotation_files(archive.namelist()):
            images: Mapping[str, dict[str, object]] = json.loads(archive.read(name))
            for item in images.values():
                user = user_index.setdefault(str(item['user_id']), len(user_index))
                hands = item['hand_landmarks']
                gestures = item['labels']
                if not isinstance(hands, list) or not isinstance(gestures, list):
                    raise ValueError(f'unexpected annotation format in {name}')
                for gesture, hand in zip(gestures, hands, strict=True):
                    if len(hand) != LANDMARKS:
                        skipped += 1
                        continue
                    points.append(hand)
                    labels.append(label_index.setdefault(str(gesture), len(label_index)))
                    users.append(user)
                    splits.append(SPLITS.index(split))
            logger.info('Read %s', name)
    logger.info('Kept %s hands; skipped %s without landmarks', f'{len(points):,}', f'{skipped:,}')
    names = sorted(label_index, key=label_index.__getitem__)
    return LandmarkTable(
        points=np.array(points, dtype=np.float32),
        label_ids=np.array(labels, dtype=np.int16),
        user_ids=np.array(users, dtype=np.int32),
        split_ids=np.array(splits, dtype=np.int8),
        label_names=tuple(names),
    )


@dataclass(frozen=True)
class GestureSample:
    '''Hands drawn for one split, with their class and original HaGRID label.'''

    points: np.ndarray
    targets: np.ndarray
    labels: np.ndarray
    users: np.ndarray

    def __len__(self) -> int:
        return len(self.points)


def draw_sample(
    table: LandmarkTable,
    split: str,
    class_of: Mapping[str, int],
    hands_per_class: int,
    seed: int,
) -> GestureSample:
    '''Draw up to hands_per_class hands for each class, spread evenly over the class's labels.

    Labels that belong to no class are left out. Each class gets an equal number of hands, so
    the classes are balanced even though, for example, rock has one label and other has many.
    '''
    rng = np.random.default_rng(seed)
    in_split = table.split_ids == SPLITS.index(split)
    chosen: list[np.ndarray] = []
    for target in sorted(set(class_of.values())):
        members = [label for label, value in class_of.items() if value == target]
        pools = []
        for label in members:
            if label not in table.label_names:
                continue
            label_id = table.label_names.index(label)
            pools.append(rng.permutation(np.flatnonzero(in_split & (table.label_ids == label_id))))
        chosen.extend(_spread(pools, hands_per_class))
    rows = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)
    targets = np.array(
        [class_of[table.label_names[label]] for label in table.label_ids[rows]], dtype=np.int64
    )
    labels = np.array([table.label_names[label] for label in table.label_ids[rows]])
    return GestureSample(
        points=table.points[rows], targets=targets, labels=labels, users=table.user_ids[rows]
    )


def _spread(pools: list[np.ndarray], total: int) -> list[np.ndarray]:
    '''Take total items across pools as evenly as their sizes allow.'''
    taken = [0] * len(pools)
    remaining = total
    open_pools = [index for index, pool in enumerate(pools) if len(pool)]
    while remaining > 0 and open_pools:
        share = max(remaining // len(open_pools), 1)
        for index in list(open_pools):
            step = min(share, len(pools[index]) - taken[index], remaining)
            taken[index] += step
            remaining -= step
            if taken[index] == len(pools[index]):
                open_pools.remove(index)
            if remaining == 0:
                break
    return [pool[:count] for pool, count in zip(pools, taken, strict=True)]
