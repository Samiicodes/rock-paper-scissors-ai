// The gesture reader: the steps of src/rpsai/landmarks.py and the trained network's forward pass.
// Both are checked against web/test/fixtures/gesture_cases.json.

const WRIST = 0;
const INDEX_BASE = 5;
const MIDDLE_BASE = 9;
const PINKY_BASE = 17;
export const LANDMARKS = 21;
export const OTHER = 3;
// A decision of UNSURE means the game asks the player to show their hand again
export const UNSURE = -1;
const CLASSES = ['rock', 'paper', 'scissors', 'other'];
const MODEL_FORMAT = 1;

/** @typedef {[number, number]} Point */

/**
 * @typedef {object} GestureModel
 * @property {number} format
 * @property {string[]} classes
 * @property {number} threshold
 * @property {{weights: number[][], bias: number[]}[]} layers
 */

/**
 * Move a hand into a standard position: wrist at the origin, the base of the middle finger at
 * (0, -1), and the index finger on the positive x side.
 * @param {Point[]} points 21 landmarks scaled to [0, 1] by the image width and height
 * @param {number} aspect image height divided by width
 * @returns {Point[]}
 */
export function canonicalise(points, aspect) {
  if (points.length !== LANDMARKS) {
    throw new Error(`expected ${LANDMARKS} landmarks`);
  }
  const scaled = points.map(([x, y]) => [x, y * aspect]);
  const [wristX, wristY] = scaled[WRIST];
  const shifted = scaled.map(([x, y]) => [x - wristX, y - wristY]);
  const [axisX, axisY] = shifted[MIDDLE_BASE];
  const length = Math.max(Math.sqrt(axisX * axisX + axisY * axisY), 1e-9);
  const cos = axisX / length;
  const sin = axisY / length;
  /** @type {Point[]} */
  const rotated = shifted.map(([x, y]) => [
    (cos * y - sin * x) / length,
    -(cos * x + sin * y) / length,
  ]);
  if (rotated[INDEX_BASE][0] < rotated[PINKY_BASE][0]) {
    return rotated.map(([x, y]) => [-x, y]);
  }
  return rotated;
}

/**
 * The classifier's input: canonical landmarks without the wrist, flattened.
 * @param {Point[]} points
 * @param {number} aspect
 * @returns {number[]}
 */
export function features(points, aspect) {
  return canonicalise(points, aspect).slice(1).flat();
}

/** A trained multilayer perceptron with ReLU hidden layers and a softmax output. */
export class GestureReader {
  /** @param {GestureModel} model */
  constructor(model) {
    if (model.format !== MODEL_FORMAT) {
      throw new Error('unsupported gesture model format');
    }
    if (model.classes.join() !== CLASSES.join()) {
      throw new Error('gesture model classes do not match');
    }
    this.layers = model.layers;
    this.threshold = model.threshold;
  }

  /**
   * Probability of rock, paper, scissors, and other.
   * @param {number[]} inputs
   * @returns {number[]}
   */
  probabilities(inputs) {
    let activation = inputs;
    this.layers.forEach((layer, index) => {
      const next = layer.bias.slice();
      for (let row = 0; row < activation.length; row += 1) {
        const value = activation[row];
        const weights = layer.weights[row];
        for (let column = 0; column < next.length; column += 1) {
          next[column] += value * weights[column];
        }
      }
      activation = index < this.layers.length - 1 ? next.map((v) => Math.max(v, 0)) : next;
    });
    const top = Math.max(...activation);
    const exps = activation.map((value) => Math.exp(value - top));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / total);
  }

  /**
   * The most likely class, or UNSURE when it is less likely than the threshold.
   * @param {number[]} probabilities
   * @returns {number}
   */
  decide(probabilities) {
    let best = 0;
    for (let index = 1; index < probabilities.length; index += 1) {
      if (probabilities[index] > probabilities[best]) {
        best = index;
      }
    }
    return probabilities[best] >= this.threshold ? best : UNSURE;
  }

  /**
   * Read one hand from MediaPipe landmarks.
   * @param {Point[]} points
   * @param {number} aspect
   * @returns {{decision: number, probabilities: number[]}}
   */
  read(points, aspect) {
    const probabilities = this.probabilities(features(points, aspect));
    return { decision: this.decide(probabilities), probabilities };
  }
}

/**
 * Settle on a move from the decisions of the frames around "shoot". A move counts when it is
 * the decision of at least minFrames frames and of at least share of all frames.
 * @param {number[]} decisions one per frame, UNSURE when no hand was found
 * @param {number} [minFrames]
 * @param {number} [share]
 * @returns {number | null} the move, or null when the hand was not read clearly
 */
export function vote(decisions, minFrames = 3, share = 0.6) {
  const counts = [0, 0, 0];
  for (const decision of decisions) {
    if (decision >= 0 && decision < 3) {
      counts[decision] += 1;
    }
  }
  const move = counts.indexOf(Math.max(...counts));
  if (counts[move] >= minFrames && counts[move] >= share * decisions.length) {
    return move;
  }
  return null;
}
