// Commit and reveal: the AI publishes the SHA-256 hash of its move and a random code before the
// player moves, and the move and code afterwards. Anyone can hash the revealed text and compare,
// which shows the AI chose before it saw the player's hand.

import { MOVE_NAMES } from './moves.js';

const CODE_BYTES = 16;

/**
 * @param {Uint8Array} bytes
 * @returns {string}
 */
function toHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * The hex SHA-256 of a text, encoded as UTF-8.
 * @param {string} text
 * @returns {Promise<string>}
 */
export async function sha256Hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return toHex(new Uint8Array(digest));
}

/**
 * Seal a move. Without the random code, the three possible hashes could be tried in turn.
 * @param {number} move
 * @param {Uint8Array} [code] random bytes, generated when not given
 * @returns {Promise<{text: string, hash: string}>}
 */
export async function seal(move, code = crypto.getRandomValues(new Uint8Array(CODE_BYTES))) {
  const text = `${MOVE_NAMES[move]}:${toHex(code)}`;
  return { text, hash: await sha256Hex(text) };
}

/**
 * Check that a revealed text matches a published hash.
 * @param {string} text
 * @param {string} hash
 * @returns {Promise<boolean>}
 */
export async function verify(text, hash) {
  return (await sha256Hex(text)) === hash;
}
