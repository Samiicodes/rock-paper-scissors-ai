import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

import { ROCK } from '../src/moves.js';
import {
  chooseMove,
  EXPERT_NAMES,
  expectedScores,
  MovePredictor,
  normalise,
  UNIFORM,
} from '../src/predictor.js';

const fixtures = JSON.parse(
  readFileSync(new URL('./fixtures/predictor_cases.json', import.meta.url), 'utf-8'),
);
const shipped = JSON.parse(
  readFileSync(new URL('../models/predictor.json', import.meta.url), 'utf-8'),
);
const TOLERANCE = 1e-12;

/**
 * @param {number[]} actual
 * @param {number[]} expected
 */
function assertClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) <= TOLERANCE,
      `${value} differs from ${expected[index]} at ${index}`,
    );
  });
}

describe('parity with the Python predictor', () => {
  it('uses the same experts', () => {
    assert.deepEqual(fixtures.experts, EXPERT_NAMES);
    assert.deepEqual(shipped.experts, EXPERT_NAMES);
  });

  for (const [caseIndex, testCase] of fixtures.cases.entries()) {
    for (const game of testCase.games) {
      it(`reproduces every forecast and move: settings ${caseIndex}, ${game.name}`, () => {
        const predictor = new MovePredictor(testCase.settings);
        for (const round of game.rounds) {
          const forecast = predictor.forecast();
          assertClose(forecast.mixture, round.mixture);
          assertClose(forecast.weights, round.weights);
          assert.equal(
            chooseMove(forecast.mixture, testCase.settings.margin, round.draw),
            round.ai,
          );
          predictor.observe(round.player, round.ai);
        }
      });
    }
  }
});

describe('MovePredictor', () => {
  it('starts with no opinion', () => {
    const forecast = new MovePredictor(shipped.settings).forecast();
    assertClose(forecast.mixture, UNIFORM);
  });

  it('learns a player who always plays rock', () => {
    const predictor = new MovePredictor(shipped.settings);
    for (let round = 0; round < 10; round += 1) {
      predictor.observe(ROCK, round % 3);
    }
    const forecast = predictor.forecast();
    assert.ok(forecast.mixture[ROCK] > 0.6);
    assert.equal(chooseMove(forecast.mixture, shipped.settings.margin, 0.5), 1);
  });

  it('rejects an empty set of experts', () => {
    assert.throws(() => new MovePredictor(shipped.settings, []));
  });
});

describe('helpers', () => {
  it('normalises counts with a prior', () => {
    assertClose(normalise([2, 0, 0], 1), [0.6, 0.2, 0.2]);
  });

  it('scores each move against a forecast', () => {
    assertClose(expectedScores([1, 0, 0]), [0, 1, -1]);
  });

  it('plays at random when nothing beats the margin', () => {
    assert.equal(chooseMove(UNIFORM, 0, 0.1), 0);
    assert.equal(chooseMove(UNIFORM, 0, 0.5), 1);
    assert.equal(chooseMove(UNIFORM, 0, 0.99), 2);
  });
});
