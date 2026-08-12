import { expect, test } from "@playwright/test";

function wavTone(seconds = 1.1, frequency = 222) {
  const sampleRate = 44_100;
  const samples = Math.round(seconds * sampleRate);
  const buffer = Buffer.alloc(44 + samples * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + samples * 2, 4);
  buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(samples * 2, 40);
  for (let index = 0; index < samples; index += 1) {
    const edge = Math.min(1, index / 2_000, (samples - index) / 2_000);
    const value = Math.sin(2 * Math.PI * frequency * index / sampleRate) * edge * 0.28;
    buffer.writeInt16LE(Math.round(value * 32_767), 44 + index * 2);
  }
  return buffer;
}

test("dry vocal completes the producer journey and downloads an MP3", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("radio", { name: "Dry vocal Key required later" }).click();
  await page.getByPlaceholder("Late-night chorus").fill("Playwright take");
  await page.locator('input[type="file"]').setInputFiles({
    name: "vocal.wav",
    mimeType: "audio/wav",
    buffer: wavTone(),
  });
  await page.getByRole("button", { name: "Listen to this take" }).click();
  await expect(page.getByRole("heading", { name: "What should this performance feel like?" })).toBeVisible();
  await page.getByPlaceholder("Intimate, aching, quiet confidence — like the room disappears around the voice…").fill("intimate and clear");
  await page.getByPlaceholder("Start, e.g. 42.5").fill("0.2");
  await page.getByPlaceholder("End, e.g. 47.0").fill("0.3");
  await page.getByRole("button", { name: "Build my tuning plan" }).click();
  await expect(page.getByRole("heading", { name: "Shape the correction" })).toBeVisible();
  await page.getByRole("button", { name: "Render previews" }).click();
  await expect(page.getByRole("button", { name: "tighter" })).toBeVisible();
  await page.getByRole("button", { name: "Render final song" }).click();
  await expect(page.getByRole("heading", { name: "Your vocal still sounds like you." })).toBeVisible();
  const download = page.getByRole("link", { name: "Vocal stem corrected-vocal.mp3 MP3 · 320 kbps" });
  await expect(download).toBeVisible();
  const downloadEvent = page.waitForEvent("download");
  await download.click();
  expect((await downloadEvent).suggestedFilename()).toBe("corrected-vocal.mp3");
});

test("vocal plus backing exposes both file inputs", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "New session" }).click();
  await page.getByRole("radio", { name: "Vocal + backing Cleanest result" }).click();
  const inputs = page.locator('input[type="file"]');
  await expect(inputs).toHaveCount(2);
  await inputs.nth(0).setInputFiles({ name: "vocal.wav", mimeType: "audio/wav", buffer: wavTone() });
  await inputs.nth(1).setInputFiles({ name: "backing.wav", mimeType: "audio/wav", buffer: wavTone(1.1, 261.63) });
  await expect(page.getByRole("button", { name: "Listen to this take" })).toBeEnabled();
});
