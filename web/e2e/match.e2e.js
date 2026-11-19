// Plays the game in a real browser. Set E2E_CHANNEL=msedge or chrome to use an installed browser
// instead of the one Playwright downloads.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { chromium } from 'playwright-core';

import { sha256Hex } from '../src/commitment.js';
import { serveWeb } from './server.js';

const ROUNDS = 20;

/** @type {Awaited<ReturnType<typeof serveWeb>>} */
let site;
/** @type {import('playwright-core').Browser} */
let browser;

before(async () => {
  site = await serveWeb();
  browser = await chromium.launch({
    channel: process.env.E2E_CHANNEL || undefined,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
  });
});

after(async () => {
  await browser?.close();
  await site?.close();
});

describe('the game in a browser', () => {
  it('plays a whole match with the keyboard and every reveal matches its seal', async () => {
    const page = await browser.newPage();
    /** @type {string[]} */
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(site.url);
    await page.getByRole('button', { name: 'Play with buttons' }).click();

    for (let round = 1; round <= ROUNDS; round += 1) {
      const fingerprint = page.locator('#ai-card .fingerprint');
      const sealText = (await fingerprint.getAttribute('title')) ?? '';
      const seal = round === 1 ? await fingerprint.textContent() : sealText;
      assert.match(seal ?? '', /^[0-9a-f]{64}$/);
      await page.keyboard.press(['r', 'p', 's', 'p'][round % 4]);
      await page
        .locator('#history li')
        .nth(round - 1)
        .waitFor();
      const reveal = (await page.locator('#reveal').textContent()) ?? '';
      const text = reveal.match(/Revealed text: (\S+)\./)?.[1] ?? '';
      assert.equal(await sha256Hex(text), seal, `round ${round}`);
      assert.match(reveal, /matches the fingerprint/);
    }

    await page.locator('#summary').waitFor({ state: 'visible' });
    const score = (await page.locator('#summary-score').textContent()) ?? '';
    const numbers = (score.match(/\d+/g) ?? []).map(Number);
    assert.equal(
      numbers.reduce((sum, value) => sum + value, 0),
      ROUNDS,
    );
    assert.deepEqual(errors, []);
    await page.close();
  });

  it('starts the camera and runs the hand tracker', async () => {
    const page = await browser.newPage();
    /** @type {string[]} */
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(site.url);
    await page.getByRole('button', { name: 'Play with your camera' }).click();
    // The fake camera shows a test pattern, so the tracker reports that it sees no hand
    await page.locator('#reading', { hasText: 'No hand in view' }).waitFor({ timeout: 60_000 });
    assert.deepEqual(errors, []);
    await page.close();
  });
});
