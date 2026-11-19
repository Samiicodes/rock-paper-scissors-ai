// The move predictor, a line-by-line port of src/rpsai/predictors.py. Both are checked against
// web/test/fixtures/predictor_cases.json, so a change here must be made there too.

import { mod3, outcome } from './moves.js';

/** @typedef {[number, number, number]} Distribution */

/**
 * @typedef {object} PredictorSettings
 * @property {number} decay
 * @property {number} prior
 * @property {number} learning_rate
 * @property {number} score_decay
 * @property {number} pattern_length
 * @property {number} margin
 */

/**
 * @typedef {object} Expert
 * @property {string} name
 * @property {() => Distribution} predict
 * @property {(player: number, ai: number) => void} observe
 */

/**
 * @typedef {object} Forecast
 * @property {Distribution} mixture
 * @property {string[]} names
 * @property {Distribution[]} predictions
 * @property {number[]} weights
 */

/** @type {Distribution} */
export const UNIFORM = [1 / 3, 1 / 3, 1 / 3];
export const TIE_TOLERANCE = 1e-12;
export const EXPERT_NAMES = [
  'uniform',
  'frequency',
  'cycle',
  'after_result',
  'markov1',
  'markov2',
  'pattern',
];

/**
 * Turn move counts into probabilities, adding the prior to each move.
 * @param {ArrayLike<number>} counts
 * @param {number} prior
 * @returns {Distribution}
 */
export function normalise(counts, prior) {
  const total = counts[0] + counts[1] + counts[2] + 3 * prior;
  return [(counts[0] + prior) / total, (counts[1] + prior) / total, (counts[2] + prior) / total];
}

/**
 * @param {number[]} row
 * @param {number} index
 * @param {number} decay
 */
function add(row, index, decay) {
  for (let position = 0; position < 3; position += 1) {
    row[position] *= decay;
  }
  row[index] += 1;
}

/** @returns {Expert} */
function uniformExpert() {
  return { name: 'uniform', predict: () => UNIFORM, observe: () => {} };
}

/**
 * @param {number} decay
 * @param {number} prior
 * @returns {Expert}
 */
function frequencyExpert(decay, prior) {
  const counts = [0, 0, 0];
  return {
    name: 'frequency',
    predict: () => normalise(counts, prior),
    observe: (player) => add(counts, player, decay),
  };
}

/**
 * @param {number} order
 * @param {number} decay
 * @param {number} prior
 * @returns {Expert}
 */
function markovExpert(order, decay, prior) {
  /** @type {number[]} */
  const history = [];
  /** @type {Map<string, number[]>} */
  const table = new Map();
  const context = () => history.slice(history.length - order).join(',');
  return {
    name: `markov${order}`,
    predict: () => {
      if (history.length < order) {
        return UNIFORM;
      }
      return normalise(table.get(context()) ?? [0, 0, 0], prior);
    },
    observe: (player) => {
      if (history.length >= order) {
        const key = context();
        let row = table.get(key);
        if (row === undefined) {
          row = [0, 0, 0];
          table.set(key, row);
        }
        add(row, player, decay);
      }
      history.push(player);
    },
  };
}

/**
 * Steps from the player's last move, counted separately after each result when byResult is set.
 * @param {string} name
 * @param {boolean} byResult
 * @param {number} decay
 * @param {number} prior
 * @returns {Expert}
 */
function shiftExpert(name, byResult, decay, prior) {
  /** @type {[number, number] | null} */
  let last = null;
  const table = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  /**
   * @param {number} player
   * @param {number} ai
   */
  const row = (player, ai) => table[byResult ? outcome(player, ai) + 1 : 0];
  return {
    name,
    predict: () => {
      if (last === null) {
        return UNIFORM;
      }
      const [player, ai] = last;
      const steps = normalise(row(player, ai), prior);
      return [steps[mod3(0 - player)], steps[mod3(1 - player)], steps[mod3(2 - player)]];
    },
    observe: (player, ai) => {
      if (last !== null) {
        const [lastPlayer, lastAi] = last;
        add(row(lastPlayer, lastAi), mod3(player - lastPlayer), decay);
      }
      last = [player, ai];
    },
  };
}

/**
 * @param {number} maxLength
 * @param {number} prior
 * @returns {Expert}
 */
function patternExpert(maxLength, prior) {
  /** @type {number[]} */
  const history = [];
  return {
    name: 'pattern',
    predict: () => {
      const size = history.length;
      for (let length = Math.min(maxLength, size - 1); length > 0; length -= 1) {
        const counts = [0, 0, 0];
        let found = false;
        for (let end = length; end < size; end += 1) {
          let matches = true;
          for (let offset = 1; offset <= length; offset += 1) {
            if (history[end - offset] !== history[size - offset]) {
              matches = false;
              break;
            }
          }
          if (matches) {
            counts[Math.floor(history[end] / 3)] += 1;
            found = true;
          }
        }
        if (found) {
          return normalise(counts, prior);
        }
      }
      return UNIFORM;
    },
    observe: (player, ai) => {
      history.push(player * 3 + ai);
    },
  };
}

/**
 * One of every expert, in the order of EXPERT_NAMES.
 * @param {PredictorSettings} settings
 * @returns {Expert[]}
 */
export function buildExperts(settings) {
  const { decay, prior } = settings;
  return [
    uniformExpert(),
    frequencyExpert(decay, prior),
    shiftExpert('cycle', false, decay, prior),
    shiftExpert('after_result', true, decay, prior),
    markovExpert(1, decay, prior),
    markovExpert(2, decay, prior),
    patternExpert(settings.pattern_length, prior),
  ];
}

/** Combines the experts with exponential weights on their recent log loss. */
export class MovePredictor {
  /**
   * @param {PredictorSettings} settings
   * @param {string[]} [experts]
   */
  constructor(settings, experts = EXPERT_NAMES) {
    this.settings = settings;
    this.experts = buildExperts(settings).filter((expert) => experts.includes(expert.name));
    if (this.experts.length === 0) {
      throw new Error('at least one expert is needed');
    }
    this.losses = this.experts.map(() => 0);
    this.rounds = 0;
  }

  /** @returns {Forecast} */
  forecast() {
    const predictions = this.experts.map((expert) => expert.predict());
    const best = Math.min(...this.losses);
    const raw = this.losses.map((loss) => Math.exp(-this.settings.learning_rate * (loss - best)));
    let total = 0;
    for (const value of raw) {
      total += value;
    }
    const weights = raw.map((value) => value / total);
    /** @type {Distribution} */
    const mixture = [0, 0, 0];
    weights.forEach((weight, index) => {
      for (let move = 0; move < 3; move += 1) {
        mixture[move] += weight * predictions[index][move];
      }
    });
    return { mixture, names: this.experts.map((expert) => expert.name), predictions, weights };
  }

  /**
   * Score every expert on the finished round, then let each learn from it.
   * @param {number} player
   * @param {number} ai
   */
  observe(player, ai) {
    const decay = this.settings.score_decay;
    this.experts.forEach((expert, index) => {
      const probability = expert.predict()[player];
      this.losses[index] = decay * this.losses[index] - Math.log(probability);
    });
    for (const expert of this.experts) {
      expert.observe(player, ai);
    }
    this.rounds += 1;
  }
}

/**
 * Expected score of each AI move against the distribution: 1 for a win, -1 for a loss.
 * @param {Distribution} distribution
 * @returns {Distribution}
 */
export function expectedScores(distribution) {
  return [
    distribution[2] - distribution[1],
    distribution[0] - distribution[2],
    distribution[1] - distribution[0],
  ];
}

/**
 * Pick the AI's move. draw is a uniform random number in [0, 1).
 * @param {Distribution} distribution
 * @param {number} margin
 * @param {number} draw
 * @returns {number}
 */
export function chooseMove(distribution, margin, draw) {
  const scores = expectedScores(distribution);
  const best = Math.max(...scores);
  if (best <= margin) {
    return Math.min(Math.floor(draw * 3), 2);
  }
  const tied = [0, 1, 2].filter((move) => scores[move] >= best - TIE_TOLERANCE);
  return tied[Math.min(Math.floor(draw * tied.length), tied.length - 1)];
}
