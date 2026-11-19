// Captures the README screenshot: a match in progress, played with the keyboard.
// Usage: node web/e2e/screenshots.js docs/figures. Set E2E_CHANNEL=msedge or chrome to use an
// installed browser.

import { join } from 'node:path';
import { chromium } from 'playwright-core';

import { serveWeb } from './server.js';

const directory = process.argv[2] ?? 'docs/figures';
// A player who mostly cycles through the moves, which the AI picks up within a few rounds
const MOVES = ['r', 'p', 's', 'r', 'p', 's', 'r', 'p'];

const site = await serveWeb();
const browser = await chromium.launch({ channel: process.env.E2E_CHANNEL || undefined });
try {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 900 },
    deviceScaleFactor: 2,
    colorScheme: 'light',
  });
  await page.goto(site.url);
  await page.getByRole('button', { name: 'Play with buttons' }).click();
  for (const [index, key] of MOVES.entries()) {
    await page.keyboard.press(key);
    await page.locator('#history li').nth(index).waitFor();
  }
  // Let the reveal animation finish
  await page.waitForTimeout(500);
  // From the top of the board to the bottom of the round history, with a margin
  const top = await page.locator('#game').boundingBox();
  const bottom = await page.locator('#history').boundingBox();
  if (top === null || bottom === null) {
    throw new Error('the game is not visible');
  }
  const margin = 20;
  const path = join(directory, 'game.png');
  await page.screenshot({
    path,
    clip: {
      x: top.x - margin,
      y: top.y - margin,
      width: top.width + 2 * margin,
      height: bottom.y + bottom.height - top.y + 2 * margin,
    },
  });
  console.log(`Saved ${path}`);
} finally {
  await browser.close();
  await site.close();
}
