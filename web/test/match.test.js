import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

import { seal, sha256Hex, verify } from '../src/commitment.js';
import { explain, strongestExpert, strongestHabit } from '../src/explain.js';
import { Match } from '../src/match.js';
import { PAPER, ROCK, SCISSORS } from '../src/moves.js';
import { MovePredictor } from '../src/predictor.js';

const { settings } = JSON.parse(
  readFileSync(new URL('../models/predictor.json', import.meta.url), 'utf-8'),
);

/**
 * A repeatable stand-in for Math.random.
 * @param {number} seed
 */
function seededRandom(seed) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

describe('commitment', () => {
  it('hashes text as SHA-256', async () => {
    assert.equal(
      await sha256Hex('abc'),
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
  });

  it('seals a move so that the reveal verifies', async () => {
    const { text, hash } = await seal(PAPER, new Uint8Array(16));
    assert.equal(text, `paper:${'00'.repeat(16)}`);
    assert.ok(await verify(text, hash));
    assert.equal(await verify('rock:00', hash), false);
  });
});

describe('Match', () => {
  it('keeps the same seal until the round is played', async () => {
    const match = new Match(settings, { random: seededRandom(1) });
    const first = await match.startRound();
    assert.equal(await match.startRound(), first);
    const record = match.play(ROCK);
    assert.equal(record.hash, first);
    assert.ok(await verify(record.reveal, record.hash));
    assert.ok(record.reveal.startsWith(['rock', 'paper', 'scissors'][record.ai]));
  });

  it('refuses to play before the AI has sealed its move', () => {
    assert.throws(() => new Match(settings).play(ROCK));
  });

  it('scores rounds and ends after the set number', async () => {
    const match = new Match(settings, { rounds: 3, random: seededRandom(2) });
    for (const move of [ROCK, PAPER, SCISSORS]) {
      await match.startRound();
      match.play(move);
    }
    const { player, ai, draws } = match.score;
    assert.equal(player + ai + draws, 3);
    assert.ok(match.finished);
    await assert.rejects(match.startRound());
    assert.equal(match.summary().rounds, 3);
  });

  it('beats a player who always shows rock', async () => {
    const match = new Match(settings, { rounds: 30, random: seededRandom(3) });
    while (!match.finished) {
      await match.startRound();
      match.play(ROCK);
    }
    const summary = match.summary();
    assert.ok(summary.aiWins >= 25, `the AI won only ${summary.aiWins} rounds`);
    assert.ok(summary.predicted >= 25);
    assert.equal(summary.habit.name, 'Favourite move');
  });
});

describe('explanations', () => {
  it('says when the AI chose at random', () => {
    const forecast = new MovePredictor(settings).forecast();
    assert.match(explain(forecast, [], true), /at random/);
    assert.equal(strongestExpert(forecast), null);
    assert.equal(strongestHabit(forecast).name, 'No pattern');
  });

  it('names the move the strongest expert expected', () => {
    const predictor = new MovePredictor(settings);
    /** @type {{player: number, ai: number}[]} */
    const history = [];
    for (let round = 0; round < 12; round += 1) {
      const player = round % 3;
      const ai = (round + 2) % 3;
      predictor.observe(player, ai);
      history.push({ player, ai });
    }
    const sentence = explain(predictor.forecast(), history, false);
    assert.match(sentence, /rock|paper|scissors/);
  });
});
