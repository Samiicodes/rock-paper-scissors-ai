import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

import { canonicalise, features, GestureReader, UNSURE, vote } from '../src/gesture.js';

/**
 * @typedef {object} GestureCase
 * @property {[number, number][]} points
 * @property {number} aspect
 * @property {number[]} features
 * @property {number[]} probabilities
 * @property {number} decision
 */

/** @type {{threshold: number, cases: GestureCase[]}} */
const fixtures = JSON.parse(
  readFileSync(new URL('./fixtures/gesture_cases.json', import.meta.url), 'utf-8'),
);
const model = JSON.parse(
  readFileSync(new URL('../models/gesture_reader.json', import.meta.url), 'utf-8'),
);
// NumPy sums matrix products in a different order, so the last bits can differ
const TOLERANCE = 1e-9;

/**
 * @param {number[]} actual
 * @param {number[]} expected
 */
function assertClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) <= TOLERANCE, `index ${index}: ${value}`);
  });
}

/** A hand pointing straight up from the wrist, with the index finger on the right. */
function uprightHand() {
  /** @type {[number, number][]} */
  const points = Array.from({ length: 21 }, (_, index) => [
    0.5 + 0.001 * index,
    0.5 - 0.01 * index,
  ]);
  points[0] = [0.5, 0.8];
  points[5] = [0.55, 0.6];
  points[9] = [0.5, 0.6];
  points[17] = [0.42, 0.62];
  return points;
}

describe('parity with the Python gesture reader', () => {
  const reader = new GestureReader(model);

  it('ships the threshold the fixtures were made with', () => {
    assert.equal(reader.threshold, fixtures.threshold);
  });

  fixtures.cases.forEach((testCase, index) => {
    it(`reads hand ${index} as Python does`, () => {
      const inputs = features(testCase.points, testCase.aspect);
      assertClose(inputs, testCase.features);
      const probabilities = reader.probabilities(inputs);
      assertClose(probabilities, testCase.probabilities);
      assert.equal(reader.decide(probabilities), testCase.decision);
    });
  });
});

describe('canonicalise', () => {
  it('puts the wrist at the origin and the middle finger base at (0, -1)', () => {
    const canonical = canonicalise(uprightHand(), 1);
    assertClose(canonical[0], [0, 0]);
    assertClose(canonical[9], [0, -1]);
  });

  it('ignores mirroring', () => {
    const hand = uprightHand();
    /** @type {[number, number][]} */
    const mirrored = hand.map(([x, y]) => [1 - x, y]);
    assertClose(features(mirrored, 1), features(hand, 1));
  });

  it('rejects a hand with the wrong number of landmarks', () => {
    assert.throws(() => canonicalise(uprightHand().slice(1), 1));
  });
});

describe('GestureReader', () => {
  it('rejects a model in another format', () => {
    assert.throws(() => new GestureReader({ ...model, format: 99 }));
    assert.throws(() => new GestureReader({ ...model, classes: ['a', 'b', 'c', 'd'] }));
  });

  it('is unsure below the threshold', () => {
    const reader = new GestureReader({ ...model, threshold: 0.9 });
    assert.equal(reader.decide([0.5, 0.3, 0.1, 0.1]), UNSURE);
    assert.equal(reader.decide([0.05, 0.92, 0.02, 0.01]), 1);
  });
});

describe('vote', () => {
  it('settles on a move most frames agree on', () => {
    assert.equal(vote([0, 0, 0, UNSURE, 0]), 0);
  });

  it('needs enough frames', () => {
    assert.equal(vote([2, 2]), null);
  });

  it('needs most frames to agree', () => {
    assert.equal(vote([1, 1, 1, 2, 2, 2, UNSURE]), null);
  });

  it('does not count non-moves', () => {
    assert.equal(vote([3, 3, 3, 3, 3]), null);
  });
});
