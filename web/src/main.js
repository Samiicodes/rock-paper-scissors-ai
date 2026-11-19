// The page: starting a match, the countdown, reading the hand, and showing each round.

import { HandCamera } from './camera.js';
import { verify } from './commitment.js';
import { GestureReader, OTHER, UNSURE, vote } from './gesture.js';
import { Match } from './match.js';
import { MOVE_NAMES } from './moves.js';

/** @typedef {import('./match.js').RoundRecord} RoundRecord */
/** @typedef {import('./gesture.js').Point} Point */

const COUNTDOWN = ['Rock', 'Paper', 'Scissors', 'Shoot!'];
const COUNT_STEP_MS = 450;
// Frames read after "Shoot!" are pooled, so one blurred frame cannot decide the move
const CAPTURE_MS = 700;
const SUMMARY_DELAY_MS = 1200;
const KEYS = /** @type {Record<string, number>} */ ({ r: 0, p: 1, s: 2 });
const LABELS = ['Rock', 'Paper', 'Scissors'];

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @returns {T}
 */
function element(id) {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`missing #${id}`);
  }
  return /** @type {T} */ (found);
}

const ui = {
  intro: element('intro'),
  introNote: element('intro-note'),
  playCamera: /** @type {HTMLButtonElement} */ (element('play-camera')),
  playButtons: /** @type {HTMLButtonElement} */ (element('play-buttons')),
  game: element('game'),
  camera: element('camera'),
  video: /** @type {HTMLVideoElement} */ (element('video')),
  overlay: /** @type {HTMLCanvasElement} */ (element('overlay')),
  reading: element('reading'),
  useButtons: /** @type {HTMLButtonElement} */ (element('use-buttons')),
  choices: element('choices'),
  roundLabel: element('round-label'),
  stage: element('stage'),
  faceoff: element('faceoff'),
  faceoffPlayer: element('faceoff-player'),
  faceoffAi: element('faceoff-ai'),
  action: /** @type {HTMLButtonElement} */ (element('action')),
  aiCard: element('ai-card'),
  scorePlayer: element('score-player'),
  scoreDraws: element('score-draws'),
  scoreAi: element('score-ai'),
  forecast: element('forecast'),
  predicted: element('predicted'),
  why: element('why'),
  history: element('history'),
  reveal: element('reveal'),
  summary: element('summary'),
  summaryScore: element('summary-score'),
  summaryRate: element('summary-rate'),
  summaryHabit: element('summary-habit'),
  playAgain: /** @type {HTMLButtonElement} */ (element('play-again')),
};

const state = {
  /** @type {import('./predictor.js').PredictorSettings | null} */
  settings: null,
  /** @type {Match | null} */
  match: null,
  /** @type {'buttons' | 'camera'} */
  mode: 'buttons',
  /** @type {HandCamera | null} */
  camera: null,
  /** @type {GestureReader | null} */
  reader: null,
  /** @type {number[] | null} decisions collected while capturing */
  frames: null,
  busy: false,
};

/**
 * @param {string} path
 * @returns {Promise<any>}
 */
async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`could not load ${path}`);
  }
  return response.json();
}

/**
 * @param {number} ms
 * @returns {Promise<void>}
 */
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * An icon for a move.
 * @param {number} move
 * @param {string} [className]
 * @returns {SVGSVGElement}
 */
function icon(move, className = '') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  if (className) {
    svg.setAttribute('class', className);
  }
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#icon-${MOVE_NAMES[move]}`);
  svg.append(use);
  return svg;
}

/**
 * @param {string} text
 * @param {'' | 'win' | 'loss' | 'small'} [tone]
 */
function setStage(text, tone = '') {
  ui.stage.textContent = text;
  ui.stage.className = tone ? `stage ${tone}` : 'stage';
}

/** @param {string} hash */
function showSealed(hash) {
  ui.aiCard.classList.remove('revealed');
  ui.aiCard.replaceChildren();
  const lock = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  lock.setAttribute('class', 'lock');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#icon-lock');
  lock.append(use);
  const title = document.createElement('span');
  title.className = 'ai-card-title';
  title.textContent = 'Move sealed';
  const fingerprint = document.createElement('code');
  fingerprint.className = 'fingerprint';
  fingerprint.textContent = hash;
  fingerprint.title = 'SHA-256 of the move and a random code';
  ui.aiCard.append(lock, title, fingerprint);
}

/**
 * @param {RoundRecord} record
 * @param {string | null} nextHash the seal of the next round, when it is already chosen
 */
function showRevealed(record, nextHash) {
  ui.aiCard.replaceChildren();
  ui.aiCard.classList.remove('revealed');
  // Restart the reveal animation
  void ui.aiCard.offsetWidth;
  ui.aiCard.classList.add('revealed');
  const name = document.createElement('span');
  name.className = 'ai-card-title';
  name.textContent = LABELS[record.ai];
  ui.aiCard.append(icon(record.ai, 'move-icon'), name);
  if (nextHash !== null) {
    const next = document.createElement('code');
    next.className = 'fingerprint';
    next.textContent = `Next move sealed: ${nextHash.slice(0, 16)}...`;
    next.title = nextHash;
    ui.aiCard.append(next);
  }
}

/** @param {RoundRecord} record */
function showForecast(record) {
  ui.forecast.replaceChildren();
  record.forecast.mixture.forEach((probability, move) => {
    const row = document.createElement('div');
    row.className = move === record.player ? 'bar-row played' : 'bar-row';
    const label = document.createElement('span');
    label.className = 'bar-label';
    label.textContent = LABELS[move];
    if (move === record.player) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = 'you';
      label.append(' ', tag);
    }
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = `${(100 * probability).toFixed(1)}%`;
    track.append(fill);
    const value = document.createElement('span');
    value.className = 'bar-value';
    value.textContent = `${Math.round(100 * probability)}%`;
    row.append(label, track, value);
    ui.forecast.append(row);
  });
}

/** @param {RoundRecord} record */
function addHistory(record) {
  const item = document.createElement('li');
  const result = record.result === 1 ? 'win' : record.result === -1 ? 'loss' : 'draw';
  item.className = result;
  const verdict = result === 'win' ? 'you won' : result === 'loss' ? 'the AI won' : 'a draw';
  item.title = `Round ${record.number}: you played ${MOVE_NAMES[record.player]}, the AI played ${MOVE_NAMES[record.ai]}, ${verdict}`;
  item.setAttribute('aria-label', item.title);
  item.append(icon(record.player, 'mine'), icon(record.ai, 'theirs'));
  ui.history.append(item);
}

/** @param {RoundRecord} record */
async function showReveal(record) {
  const matches = await verify(record.reveal, record.hash);
  ui.reveal.textContent = `Round ${record.number}. Fingerprint shown before you played: ${record.hash}. Revealed text: ${record.reveal}. SHA-256 of the revealed text ${matches ? 'matches' : 'does not match'} the fingerprint.`;
}

function updateScore() {
  const match = /** @type {Match} */ (state.match);
  const { player, ai, draws } = match.score;
  ui.scorePlayer.textContent = String(player);
  ui.scoreAi.textContent = String(ai);
  ui.scoreDraws.textContent = String(draws);
  const played = match.history.length;
  const predicted = match.history.filter((round) => round.predicted).length;
  const rate = Math.round((100 * ai) / Math.max(played, 1));
  ui.predicted.textContent = `So far I have predicted your move in ${predicted} of ${played} rounds and won ${rate}% of rounds. Random play wins about 33%.`;
}

function updateRoundLabel() {
  const match = /** @type {Match} */ (state.match);
  const round = Math.min(match.history.length + 1, match.rounds);
  ui.roundLabel.textContent = `Round ${round} of ${match.rounds}`;
}

/** @param {boolean} enabled */
function setChoices(enabled) {
  for (const button of ui.choices.querySelectorAll('button')) {
    button.disabled = !enabled;
  }
}

/** Seal the AI's next move and get the board ready for the player. */
async function beginRound() {
  const match = /** @type {Match} */ (state.match);
  const hash = await match.startRound();
  updateRoundLabel();
  if (match.history.length === 0) {
    showSealed(hash);
  }
  if (state.mode === 'buttons') {
    setChoices(true);
    if (match.history.length === 0) {
      setStage('Pick your move', 'small');
    }
  } else {
    ui.action.hidden = false;
    ui.action.disabled = false;
    ui.action.textContent = match.history.length === 0 ? 'Play round' : 'Play next round';
    if (match.history.length === 0) {
      setStage('Press play, then show your hand on "Shoot!"', 'small');
    }
  }
  return hash;
}

/**
 * Play the player's move and show the round.
 * @param {number} move
 */
async function finishRound(move) {
  const match = /** @type {Match} */ (state.match);
  const record = match.play(move);
  const tone = record.result === 1 ? 'win' : record.result === -1 ? 'loss' : '';
  setStage(record.result === 1 ? 'You win' : record.result === -1 ? 'I win' : 'Draw', tone);
  ui.faceoff.hidden = false;
  ui.faceoffPlayer.replaceChildren(icon(record.player), `You: ${LABELS[record.player]}`);
  ui.faceoffAi.replaceChildren(icon(record.ai), `AI: ${LABELS[record.ai]}`);
  showForecast(record);
  ui.why.textContent = record.explanation;
  addHistory(record);
  updateScore();
  await showReveal(record);
  if (match.finished) {
    showRevealed(record, null);
    ui.action.hidden = true;
    setChoices(false);
    await wait(SUMMARY_DELAY_MS);
    showSummary();
    return;
  }
  const nextHash = await beginRound();
  showRevealed(record, nextHash);
}

/** @param {number} move */
async function chooseWithButton(move) {
  if (state.busy || state.mode !== 'buttons' || state.match === null || state.match.finished) {
    return;
  }
  state.busy = true;
  setChoices(false);
  try {
    await finishRound(move);
  } finally {
    state.busy = false;
  }
}

/** Count down, read the hand around "Shoot!", and play it. */
async function playWithCamera() {
  const match = /** @type {Match} */ (state.match);
  if (state.busy || match.finished) {
    return;
  }
  state.busy = true;
  ui.action.disabled = true;
  try {
    showSealed(await match.startRound());
    for (const word of COUNTDOWN.slice(0, -1)) {
      setStage(word);
      await wait(COUNT_STEP_MS);
    }
    setStage(COUNTDOWN[COUNTDOWN.length - 1]);
    state.frames = [];
    await wait(CAPTURE_MS);
    const move = vote(state.frames);
    state.frames = null;
    if (move === null) {
      setStage(
        'I could not read your hand. Hold rock, paper, or scissors up to the camera.',
        'small',
      );
      ui.action.textContent = 'Try again';
      ui.action.disabled = false;
      return;
    }
    await finishRound(move);
  } finally {
    state.busy = false;
  }
}

/**
 * Show the live reading under the video and collect it while capturing.
 * @param {Point[] | null} points
 * @param {number} aspect
 */
function onFrame(points, aspect) {
  const reader = /** @type {GestureReader} */ (state.reader);
  let decision = UNSURE;
  if (points === null) {
    ui.reading.textContent = 'No hand in view';
  } else {
    const reading = reader.read(points, aspect);
    decision = reading.decision;
    if (decision === UNSURE) {
      ui.reading.textContent = 'Not sure';
    } else if (decision === OTHER) {
      ui.reading.textContent = 'Not a move';
    } else {
      ui.reading.textContent = `${LABELS[decision]} ${Math.round(100 * reading.probabilities[decision])}%`;
    }
  }
  state.frames?.push(decision);
}

/** @returns {Promise<boolean>} whether the camera started */
async function startCamera() {
  ui.introNote.textContent = 'Starting the camera and loading the hand tracker...';
  ui.introNote.classList.remove('error');
  try {
    state.reader ??= new GestureReader(await loadJson('models/gesture_reader.json'));
    state.camera = await HandCamera.open(ui.video, ui.overlay);
    state.camera.start(onFrame);
    return true;
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    ui.introNote.textContent = `The camera could not start (${reason}). You can still play with buttons.`;
    ui.introNote.classList.add('error');
    return false;
  }
}

/** @param {'buttons' | 'camera'} mode */
function setMode(mode) {
  state.mode = mode;
  const camera = mode === 'camera';
  ui.camera.hidden = !camera;
  ui.useButtons.hidden = !camera;
  ui.choices.hidden = camera;
  ui.action.hidden = !camera;
}

/** @param {'buttons' | 'camera'} mode */
async function startGame(mode) {
  if (state.busy) {
    return;
  }
  state.busy = true;
  try {
    // A click can arrive before the settings have downloaded
    await settingsLoaded;
    if (state.settings === null) {
      return;
    }
    if (mode === 'camera' && !(await startCamera())) {
      return;
    }
    setMode(mode);
    ui.intro.hidden = true;
    ui.game.hidden = false;
    newMatch();
    await beginRound();
  } finally {
    state.busy = false;
  }
}

function newMatch() {
  state.match = new Match(
    /** @type {import('./predictor.js').PredictorSettings} */ (state.settings),
  );
  ui.history.replaceChildren();
  ui.forecast.replaceChildren();
  ui.faceoff.hidden = true;
  ui.why.textContent = 'I have not seen you play yet.';
  ui.predicted.textContent = '';
  ui.reveal.textContent = 'Play a round to see a reveal.';
  updateScore();
  ui.predicted.textContent = '';
}

function showSummary() {
  const summary = /** @type {Match} */ (state.match).summary();
  const rate = Math.round((100 * summary.aiWins) / summary.rounds);
  ui.summaryScore.textContent = `You ${summary.playerWins}, AI ${summary.aiWins}, draws ${summary.draws}`;
  ui.summaryRate.textContent = `I won ${rate}% of the rounds and predicted your move in ${summary.predicted} of ${summary.rounds}. Playing at random, I would win about 33%.`;
  ui.summaryHabit.textContent = `The habit I relied on most: ${summary.habit.name.toLowerCase()}. ${summary.habit.summary}`;
  ui.summary.hidden = false;
  ui.playAgain.focus();
}

async function playAgain() {
  ui.summary.hidden = true;
  newMatch();
  setMode(state.mode);
  await beginRound();
}

const settingsLoaded = loadJson('models/predictor.json').then(
  (predictor) => {
    state.settings = predictor.settings;
  },
  () => {
    ui.introNote.textContent = 'The game could not load. Please reload the page.';
    ui.introNote.classList.add('error');
  },
);

function init() {
  ui.playCamera.addEventListener('click', () => startGame('camera'));
  ui.playButtons.addEventListener('click', () => startGame('buttons'));
  ui.action.addEventListener('click', () => playWithCamera());
  ui.playAgain.addEventListener('click', () => playAgain());
  ui.useButtons.addEventListener('click', () => {
    state.camera?.stop();
    state.camera = null;
    setMode('buttons');
    setChoices(true);
    setStage('Pick your move', 'small');
  });
  for (const button of ui.choices.querySelectorAll('button')) {
    button.addEventListener('click', () => chooseWithButton(Number(button.dataset.move)));
  }
  document.addEventListener('keydown', (event) => {
    const move = KEYS[event.key.toLowerCase()];
    if (move !== undefined && !event.ctrlKey && !event.metaKey && !event.altKey) {
      chooseWithButton(move);
    }
  });
}

init();
