// Moves and outcomes, encoded as in src/rpsai/moves.py: each move beats the one before it.

export const ROCK = 0;
export const PAPER = 1;
export const SCISSORS = 2;
export const MOVE_NAMES = /** @type {const} */ (['rock', 'paper', 'scissors']);

/**
 * The move that beats the given move.
 * @param {number} move
 * @returns {number}
 */
export function counter(move) {
  return (move + 1) % 3;
}

/**
 * 1 when the player wins, 0 for a draw, and -1 when the player loses.
 * @param {number} player
 * @param {number} opponent
 * @returns {number}
 */
export function outcome(player, opponent) {
  if (player === opponent) {
    return 0;
  }
  return player === counter(opponent) ? 1 : -1;
}

/**
 * The remainder of value divided by 3, always between 0 and 2.
 * @param {number} value
 * @returns {number}
 */
export function mod3(value) {
  return ((value % 3) + 3) % 3;
}
