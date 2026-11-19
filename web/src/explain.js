// Plain-language reasons for the AI's forecast, taken from the expert it trusted most.

import { counter, MOVE_NAMES, mod3, outcome } from './moves.js';

/** @typedef {import('./predictor.js').Forecast} Forecast */
/** @typedef {{player: number, ai: number}} Round */

export const HABIT_NAMES = /** @type {Record<string, string>} */ ({
  uniform: 'No pattern',
  frequency: 'Favourite move',
  cycle: 'Cycling',
  after_result: 'After a win or loss',
  markov1: 'After the last move',
  markov2: 'After the last two moves',
  pattern: 'Repeated sequence',
});

const HABIT_SUMMARIES = /** @type {Record<string, string>} */ ({
  uniform: 'I could not find a habit that held up. Well played.',
  frequency: 'You leaned on one move more than the others.',
  cycle: 'You often stepped through the moves in order.',
  after_result: 'Winning or losing changed what you played next.',
  markov1: 'Your next move often depended on your last one.',
  markov2: 'Your next move often depended on your last two.',
  pattern: 'You repeated whole sequences of rounds.',
});

const RESULT_PHRASES = { '-1': 'a loss', 0: 'a draw', 1: 'a win' };

/**
 * The expert with the largest weight, if it outweighs the no-pattern expert.
 * @param {Forecast} forecast
 * @returns {number | null} its index in the forecast, or null when no pattern stands out
 */
export function strongestExpert(forecast) {
  let best = -1;
  let uniformWeight = 0;
  forecast.names.forEach((name, index) => {
    if (name === 'uniform') {
      uniformWeight = forecast.weights[index];
    } else if (best < 0 || forecast.weights[index] > forecast.weights[best]) {
      best = index;
    }
  });
  return best >= 0 && forecast.weights[best] > uniformWeight ? best : null;
}

/**
 * @param {number[]} distribution
 * @returns {number}
 */
function likeliest(distribution) {
  return distribution.indexOf(Math.max(...distribution));
}

/**
 * Explain the forecast made before a round, using the rounds that came before it.
 * @param {Forecast} forecast
 * @param {Round[]} history
 * @param {boolean} random whether the AI ended up choosing at random
 * @returns {string}
 */
export function explain(forecast, history, random) {
  const index = strongestExpert(forecast);
  if (random || index === null || history.length === 0) {
    return 'No clear pattern yet, so I chose at random.';
  }
  const name = forecast.names[index];
  const expected = likeliest(forecast.predictions[index]);
  const move = MOVE_NAMES[expected];
  const last = history[history.length - 1];
  const lastMove = MOVE_NAMES[last.player];
  switch (name) {
    case 'frequency':
      return `You have played ${move} more than anything else lately.`;
    case 'cycle': {
      const step = mod3(expected - last.player);
      if (step === 0) {
        return `You often play the same move twice in a row, so I expected ${move} again.`;
      }
      const direction = step === 1 ? 'rock, paper, scissors' : 'scissors, paper, rock';
      return `You tend to step through the moves in the order ${direction}, so after ${lastMove} I expected ${move}.`;
    }
    case 'after_result': {
      const result = RESULT_PHRASES[/** @type {-1 | 0 | 1} */ (outcome(last.player, last.ai))];
      if (expected === last.player) {
        return `After ${result} you tend to play the same move again, so I expected ${move}.`;
      }
      const relation =
        expected === counter(last.player)
          ? 'the move that beats your last one'
          : 'the move your last one beats';
      return `After ${result} you tend to switch to ${relation}, so I expected ${move}.`;
    }
    case 'markov1':
      return `After ${lastMove} you have often played ${move}.`;
    case 'markov2': {
      const before = history.length > 1 ? MOVE_NAMES[history[history.length - 2].player] : lastMove;
      return `After ${before} then ${lastMove}, you have often played ${move}.`;
    }
    case 'pattern':
      return `The last few rounds repeat an earlier part of the game, and back then you followed with ${move}.`;
    default:
      return `I expected ${move}.`;
  }
}

/**
 * Name and describe the habit the AI relied on most by the end of a match.
 * @param {Forecast} forecast
 * @returns {{name: string, summary: string}}
 */
export function strongestHabit(forecast) {
  const index = strongestExpert(forecast);
  const key = index === null ? 'uniform' : forecast.names[index];
  return { name: HABIT_NAMES[key], summary: HABIT_SUMMARIES[key] };
}
