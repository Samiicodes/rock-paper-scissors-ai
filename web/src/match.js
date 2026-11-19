// A match against the AI: sealing its move, scoring rounds, and summing up. No browser APIs, so
// it runs the same in the page and in the tests.

import { seal } from './commitment.js';
import { explain, strongestHabit } from './explain.js';
import { outcome } from './moves.js';
import { chooseMove, expectedScores, MovePredictor, TIE_TOLERANCE } from './predictor.js';

/** @typedef {import('./predictor.js').Forecast} Forecast */
/** @typedef {import('./predictor.js').PredictorSettings} PredictorSettings */

/**
 * @typedef {object} RoundRecord
 * @property {number} number
 * @property {number} player
 * @property {number} ai
 * @property {number} result 1 when the player won, 0 for a draw, -1 when the AI won
 * @property {Forecast} forecast what the AI expected before the round
 * @property {boolean} predicted whether the AI clearly favoured the move the player chose
 * @property {string} explanation
 * @property {string} hash the seal published before the round
 * @property {string} reveal the text that was sealed
 */

/**
 * @typedef {object} Summary
 * @property {number} rounds
 * @property {number} playerWins
 * @property {number} aiWins
 * @property {number} draws
 * @property {number} predicted rounds in which the AI's most likely move was the one played
 * @property {{name: string, summary: string}} habit
 */

export const MATCH_ROUNDS = 20;

/**
 * The move the forecast clearly favours, or null when two or three moves tie for most likely.
 * @param {number[]} mixture
 * @returns {number | null}
 */
export function favourite(mixture) {
  const best = Math.max(...mixture);
  const tied = [0, 1, 2].filter((move) => mixture[move] >= best - TIE_TOLERANCE);
  return tied.length === 1 ? tied[0] : null;
}

export class Match {
  /**
   * @param {PredictorSettings} settings
   * @param {{rounds?: number, random?: () => number}} [options]
   */
  constructor(settings, { rounds = MATCH_ROUNDS, random = Math.random } = {}) {
    this.settings = settings;
    this.rounds = rounds;
    this.random = random;
    this.predictor = new MovePredictor(settings);
    /** @type {RoundRecord[]} */
    this.history = [];
    /** @type {{forecast: Forecast, ai: number, random: boolean, text: string, hash: string} | null} */
    this.pending = null;
  }

  get finished() {
    return this.history.length >= this.rounds;
  }

  get score() {
    const count = (/** @type {number} */ value) =>
      this.history.filter((round) => round.result === value).length;
    return { player: count(1), ai: count(-1), draws: count(0) };
  }

  /**
   * Choose and seal the AI's move for the next round. Calling it again before play returns the
   * same seal, so the AI cannot change its mind.
   * @returns {Promise<string>} the hash to show the player
   */
  async startRound() {
    if (this.finished) {
      throw new Error('the match is over');
    }
    if (this.pending === null) {
      const forecast = this.predictor.forecast();
      const ai = chooseMove(forecast.mixture, this.settings.margin, this.random());
      const random = Math.max(...expectedScores(forecast.mixture)) <= this.settings.margin;
      const { text, hash } = await seal(ai);
      this.pending = { forecast, ai, random, text, hash };
    }
    return this.pending.hash;
  }

  /**
   * Play the player's move against the sealed AI move.
   * @param {number} player
   * @returns {RoundRecord}
   */
  play(player) {
    if (this.pending === null) {
      throw new Error('startRound must be called before play');
    }
    const { forecast, ai, random, text, hash } = this.pending;
    const mixture = forecast.mixture;
    const record = {
      number: this.history.length + 1,
      player,
      ai,
      result: outcome(player, ai),
      forecast,
      predicted: favourite(mixture) === player,
      explanation: explain(forecast, this.history, random),
      hash,
      reveal: text,
    };
    this.predictor.observe(player, ai);
    this.history.push(record);
    this.pending = null;
    return record;
  }

  /** @returns {Summary} */
  summary() {
    const { player, ai, draws } = this.score;
    return {
      rounds: this.history.length,
      playerWins: player,
      aiWins: ai,
      draws,
      predicted: this.history.filter((round) => round.predicted).length,
      habit: strongestHabit(this.predictor.forecast()),
    };
  }
}
