// Browser checks intercept every API call; they never send real inference jobs.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const baseURL = process.env.FRONTEND_URL || "http://127.0.0.1:8098";
const imageId = "f5890ad4-90a8-4c51-9b1c-e37ac49d483f";
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1cAAAAASUVORK5CYII=", "base64");

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.addInitScript(() => {
      window.savedFiles = {};
      window.showDirectoryPicker = async () => ({
        name: "Local results",
        requestPermission: async () => "granted",
        getDirectoryHandle: async (directory) => ({
          getFileHandle: async (name) => ({
            createWritable: async () => {
              const chunks = [];
              const record = () => { window.savedFiles[`${directory}/${name}`] = chunks.flat(); };
              const stream = new WritableStream({
                write(chunk) { chunks.push(Array.from(chunk)); },
                close: record,
              });
              // FileSystemWritableFileStream also has direct write/close methods.
              stream.write = async (blob) => { chunks.push(Array.from(new Uint8Array(await blob.arrayBuffer()))); };
              stream.close = async () => record();
              return stream;
            },
          }),
        }),
      });
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const requests = [];
    let ready = false;
    let uploadError = false;
    const files = {
      original: png,
      "normalized.png": png,
      "detection.png": png,
      "results.json": Buffer.from(JSON.stringify({ id: imageId, annotation: { raw_caption: "  A dog.  " } })),
    };
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      requests.push(path);
      if (path === "/api/images") {
        assert.match(route.request().headers()["content-type"], /multipart\/form-data/);
        return route.fulfill({ status: uploadError ? 503 : 202, json: { image_id: imageId, output_folder: `results/${imageId}` } });
      }
      assert(!path.includes("/jobs/"), "No polling or job API is allowed");
      if (!ready) return route.fulfill({ status: 404, json: { detail: "result_not_found" } });
      if (path.endsWith("/archive")) return route.fulfill({ contentType: "application/zip", body: Buffer.from("archive") });
      const name = path.split("/").pop();
      assert(name in files, `Unexpected API path ${path}`);
      return route.fulfill({ contentType: name.endsWith("json") ? "application/json" : "application/octet-stream", body: files[name] });
    });
    await page.goto(baseURL);
    await page.locator("#file").setInputFiles({ name: "dog.png", mimeType: "image/png", buffer: png });
    await page.locator("#submit").click();
    await page.waitForFunction(() => !document.getElementById("save-actions").hidden);
    assert.equal(requests.length, 1, "Only an upload should happen until the user requests a copy");
    assert.match(await page.locator("#status").textContent(), /volume: results/);
    await page.locator("#save-copy").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("ещё не готовы"));
    ready = true;
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#save-copy").click();
    const download = await downloadPromise;
    assert.equal(download.suggestedFilename(), `${imageId}.zip`);
    await page.locator("#choose-folder").click();
    assert.equal(await page.locator("#folder").inputValue(), "Local results");
    await page.locator("#save-copy").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("Копия сохранена"));
    const saved = await page.evaluate(() => window.savedFiles);
    assert.deepEqual(Object.keys(saved).sort(), Object.keys(files).map((name) => `${imageId}/${name}`).sort());
    for (const [name, content] of Object.entries(files)) assert.deepEqual(Buffer.from(saved[`${imageId}/${name}`]), content);
    assert.match(await page.locator("#status").textContent(), /остаётся в volume/);
    if (process.env.SCREENSHOT_DIR) await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/simple-desktop.png`, fullPage: true });
    const countBeforeResize = requests.length;
    await page.setViewportSize({ width: 375, height: 812 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    assert.equal(requests.length, countBeforeResize);
    if (process.env.SCREENSHOT_DIR) await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/simple-mobile.png`, fullPage: true });
    uploadError = true;
    await page.locator("#submit").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("не подтверждена"));
    assert(await page.locator("#save-actions").isHidden());
    assert.deepEqual(errors, []);
    const fallback = await browser.newPage();
    await fallback.addInitScript(() => { delete window.showDirectoryPicker; });
    await fallback.goto(baseURL);
    assert(await fallback.locator("#choose-folder").isHidden());
    assert.match(await fallback.locator("#folder").inputValue(), /ZIP/);
    console.log("Browser checks passed: no polling, explicit copy, four local files, ZIP, early download, upload failure, unsupported browser, mobile width.");
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
