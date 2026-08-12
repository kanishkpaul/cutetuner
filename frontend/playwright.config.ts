import { defineConfig, devices } from "@playwright/test";

const dataDir = `/tmp/cutetuner-playwright-${process.pid}`;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8766",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `CUTETUNER_LOW_MEMORY=1 CUTETUNER_DATA_DIR=${dataDir} uv run cutetuner ui --no-open --port 8766`,
    cwd: "..",
    url: "http://127.0.0.1:8766/api/health",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
