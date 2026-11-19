'''Recorded games of people playing rock, paper, scissors against a computer.

The data comes from Komai, Kurokawa, and Kim (2022): 500 people who each played 50 rounds. It is
published at github.com/kuro-lab/RPSdata without a licence, so it is downloaded for evaluation
only and never committed.
'''

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpsai.moves import Move

logger = logging.getLogger(__name__)

PLAYERS = 500
SERIES_DIR = 'Human_and_computer_hand_series_data'

# The dataset numbers moves 0, 1, 2 without documenting them. Its authors' later paper uses
# 0 rock, 1 scissors, 2 paper, and with that reading players repeat a move more after a win
# than after a loss, as earlier studies found.
SOURCE_ENCODING = {0: Move.ROCK, 1: Move.SCISSORS, 2: Move.PAPER}


@dataclass(frozen=True)
class HumanGames:
    '''Moves of each player and of the computer they played, one row per player.'''

    player_ids: np.ndarray
    player: np.ndarray
    opponent: np.ndarray

    def __post_init__(self) -> None:
        if self.player.shape != self.opponent.shape or self.player.ndim != 2:
            raise ValueError('player and opponent moves must be matching 2D arrays')
        if len(self.player_ids) != len(self.player):
            raise ValueError('one player id is needed per row')

    def __len__(self) -> int:
        return len(self.player)

    def subset(self, rows: np.ndarray) -> 'HumanGames':
        '''Return the games of the selected rows.'''
        return HumanGames(self.player_ids[rows], self.player[rows], self.opponent[rows])


def parse_series(text: str) -> list[int]:
    '''Parse one comma-separated file of moves, ignoring the trailing comma.'''
    return [SOURCE_ENCODING[int(value)] for value in text.split(',') if value.strip()]


def load_human_games(archive_path: Path) -> HumanGames:
    '''Read every complete game from the downloaded repository archive.'''
    with zipfile.ZipFile(archive_path) as archive:
        root = next(name for name in archive.namelist() if name.endswith(f'/{SERIES_DIR}/'))
        ids, players, opponents = [], [], []
        for player_id in range(1, PLAYERS + 1):
            player = parse_series(
                archive.read(f'{root}Human_hand_data/{player_id}.csv').decode('utf-8')
            )
            opponent = parse_series(
                archive.read(f'{root}AI_hand_data/{player_id}.csv').decode('utf-8')
            )
            if len(player) != len(opponent):
                logger.info(
                    'Skipped player %d: %d moves against %d computer moves',
                    player_id,
                    len(player),
                    len(opponent),
                )
                continue
            ids.append(player_id)
            players.append(player)
            opponents.append(opponent)
    return HumanGames(
        player_ids=np.array(ids),
        player=np.array(players, dtype=np.int64),
        opponent=np.array(opponents, dtype=np.int64),
    )


def split_players(games: HumanGames, fraction: float, seed: int) -> tuple[HumanGames, HumanGames]:
    '''Split players at random into a tuning set and a test set.'''
    order = np.random.default_rng(seed).permutation(len(games))
    cut = round(len(games) * fraction)
    return games.subset(np.sort(order[:cut])), games.subset(np.sort(order[cut:]))
