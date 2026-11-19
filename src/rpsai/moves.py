'''Moves and outcomes, encoded the same way as in the web game.'''

from enum import IntEnum


class Move(IntEnum):
    '''A hand in rock, paper, scissors. Each move beats the one before it, cyclically.'''

    ROCK = 0
    PAPER = 1
    SCISSORS = 2


MOVE_NAMES = tuple(move.name.lower() for move in Move)


def counter(move: int) -> int:
    '''Return the move that beats the given move.'''
    return (move + 1) % 3


def outcome(player: int, opponent: int) -> int:
    '''Return 1 when the player wins, 0 for a draw, and -1 when the player loses.'''
    if player == opponent:
        return 0
    return 1 if player == counter(opponent) else -1
