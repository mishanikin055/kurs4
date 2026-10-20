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
        queryPermission: async () => window.folderPermission || "granted",
        getDirectoryHandle: async (directory) => ({
          getFileHandle: async (name) => ({
            createWritable: async () => {
              if (window.folderWriteError) throw new Error("write failed");
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
    let resultError = false;
    const result = {
      id: imageId, width: 640, height: 480,
      detection: { objects: [{ object_id: "dog", label: "dog" }, { object_id: "cat", label: "cat" }] },
      classification: { objects: [{ object_id: "cat", label: "unicycle", refinement: null }, { object_id: "dog", label: "Labrador retriever", refinement: "Labrador retriever" }] },
      annotation: { raw_caption: "  A dog and <b>a cat</b>.  " },
    };
    let preview = result;
    const files = {
      original: png,
      "normalized.png": png,
      "detection.png": png,
      "results.json": Buffer.from(JSON.stringify({
        ...result,
        classification: { objects: result.classification.objects.map(({ refinement, ...object }) => object) },
      })),
    };
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      requests.push(path);
      if (path === "/api/images") {
        assert.match(route.request().headers()["content-type"], /multipart\/form-data/);
        return route.fulfill({ status: uploadError ? 503 : 202, json: { image_id: imageId, output_folder: `results/${imageId}` } });
      }
      assert(!path.includes("/jobs/"), "No job API is allowed");
      if (resultError) return route.fulfill({ status: 503, json: { detail: "unavailable" } });
      if (!ready) return route.fulfill({ status: 404, json: { detail: "result_not_found" } });
      if (path === `/api/images/${imageId}`) return route.fulfill({ json: preview });
      if (path.endsWith("/archive")) return route.fulfill({ contentType: "application/zip", body: Buffer.from("archive") });
      const name = path.split("/").pop();
      assert(name in files, `Unexpected API path ${path}`);
      return route.fulfill({ contentType: name.endsWith("json") ? "application/json" : "application/octet-stream", body: files[name] });
    });
    await page.goto(baseURL);
    await page.locator("#file").setInputFiles({ name: "dog.png", mimeType: "image/png", buffer: png });
    await page.locator("#submit").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("автоматически"));
    assert.equal(await page.locator("#show-result").count(), 0);
    assert.equal(await page.locator("#save-copy").count(), 0);
    assert(!/volume|results\//.test(await page.locator("#status").textContent()));
    await page.waitForResponse((response) => response.url().endsWith(`/api/images/${imageId}`));
    assert(requests.filter((path) => path === `/api/images/${imageId}`).length >= 2, "Pending results are retried automatically");
    assert(await page.locator("#result").isHidden());
    assert(await page.locator("#save-actions").isHidden());
    ready = true;
    await page.locator("#result").waitFor({ state: "visible" });
    await page.waitForFunction(() => document.getElementById("result-image").naturalWidth > 0);
    assert.equal(await page.locator("#caption").textContent(), result.annotation.raw_caption);
    assert.equal(await page.locator("#caption b").count(), 0, "Model text must never become HTML");
    assert.deepEqual(await page.locator("#objects tr").allTextContents(), ["1dogLabrador retriever", "2cat"]);
    const controls = await page.locator(".controls").boundingBox();
    const image = await page.locator("#result-image").boundingBox();
    assert(image.x >= controls.x + controls.width, "Image is to the right of the controls and table");
    assert(await page.locator("#submit").evaluate((button) => button.form.id === "upload-form" && !button.closest("#upload-form") && !button.closest("#dropzone")), "Submit is outside both upload blocks and still submits the form");
    assert(await page.locator("#result").evaluate((result) => !result.closest(".controls")), "Results have their own panel");
    for (const viewport of [{ width: 1366, height: 650 }, { width: 1280, height: 600 }, { width: 1536, height: 730 }, { width: 1024, height: 600 }]) {
      await page.setViewportSize(viewport);
      assert(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight && document.documentElement.scrollWidth <= innerWidth), `No page scrolling at ${viewport.width}x${viewport.height}`);
      for (const selector of ["#submit", ".result-panel", ".preview"]) {
        const box = await page.locator(selector).boundingBox();
        assert(box.y >= 0 && box.y + box.height <= viewport.height, `${selector} fits the laptop viewport`);
      }
    }
    await page.evaluate(() => {
      const rows = document.getElementById("objects");
      for (let i = 0; i < 60; i++) rows.append(rows.firstElementChild.cloneNode(true));
    });
    assert(await page.locator("#objects-table").evaluate((table) => table.scrollHeight > table.clientHeight), "Long tables scroll within the result panel");
    assert(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight), "Long tables do not make the page scroll");
    await page.evaluate(() => {
      const rows = document.getElementById("objects");
      while (rows.children.length > 2) rows.lastElementChild.remove();
    });
    await page.setViewportSize({ width: 1440, height: 1000 });
    const countAfterResult = requests.length;
    await page.waitForTimeout(2200);
    assert.equal(requests.length, countAfterResult, "Polling stops after the result is shown");
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#download-zip").click();
    const download = await downloadPromise;
    assert.equal(download.suggestedFilename(), `${imageId}.zip`);
    await page.locator("#choose-folder").click();
    assert.equal(await page.locator("#folder").inputValue(), "Local results");
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("Копия сохранена"));
    const saved = await page.evaluate(() => window.savedFiles);
    assert.deepEqual(Object.keys(saved).sort(), Object.keys(files).map((name) => `${imageId}/${name}`).sort());
    for (const [name, content] of Object.entries(files)) assert.deepEqual(Buffer.from(saved[`${imageId}/${name}`]), content);
    assert(!/volume/.test(await page.locator("#status").textContent()));
    assert(await page.locator("#save-actions").isHidden(), "Selected folder is saved automatically without a save button");
    await page.evaluate(() => { window.savedFiles = {}; });
    await page.locator("#submit").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("Копия сохранена"));
    assert.equal(await page.evaluate(() => Object.keys(window.savedFiles).length), 4, "Folder selected before upload receives all four files automatically");
    for (const flag of ["folderPermission", "folderWriteError"]) {
      await page.evaluate((key) => { window[key] = key === "folderPermission" ? "denied" : true; }, flag);
      await page.locator("#choose-folder").click();
      await page.waitForFunction(() => document.getElementById("status").textContent.includes("сохранить полный комплект не удалось"));
      assert(await page.locator("#result").isVisible(), "Saving failure keeps the result visible");
      assert(await page.locator("#download-zip").isVisible(), "Saving failure offers ZIP");
      await page.evaluate((key) => { delete window[key]; }, flag);
    }
    await page.locator("#choose-folder").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("Копия сохранена"));
    if (process.env.SCREENSHOT_DIR) await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/simple-desktop.png`, fullPage: true });
    const countBeforeResize = requests.length;
    await page.setViewportSize({ width: 375, height: 812 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    assert.equal(requests.length, countBeforeResize);
    if (process.env.SCREENSHOT_DIR) await page.screenshot({ path: `${process.env.SCREENSHOT_DIR}/simple-mobile.png`, fullPage: true });
    preview = { ...result, detection: { objects: [] }, classification: { objects: [] } };
    await page.locator("#submit").click();
    await page.waitForFunction(() => !document.getElementById("no-objects").hidden);
    assert(await page.locator("#objects-table").isHidden());
    resultError = true;
    await page.locator("#submit").click();
    await page.locator("#retry-result").waitFor({ state: "visible" });
    assert.match(await page.locator("#status").textContent(), /Не удалось загрузить/);
    assert(await page.locator("#result").isHidden());
    const countAfterError = requests.length;
    await page.waitForTimeout(2200);
    assert.equal(requests.length, countAfterError, "Network/server error stops automatic requests");
    resultError = false;
    await page.locator("#retry-result").click();
    await page.locator("#result").waitFor({ state: "visible" });
    assert(await page.locator("#retry-result").isHidden());
    // A new upload cancels pending polling even if that upload fails.
    ready = false;
    await page.locator("#submit").click();
    await page.waitForResponse((response) => response.url().endsWith(`/api/images/${imageId}`));
    uploadError = true;
    await page.locator("#submit").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.includes("не подтверждена"));
    assert(await page.locator("#save-actions").isHidden());
    assert(await page.locator("#result").isHidden(), "New upload clears the old result");
    const countAfterReset = requests.length;
    await page.waitForTimeout(2200);
    assert.equal(requests.length, countAfterReset, "New upload cancels the previous timer");
    assert.deepEqual(errors, []);
    const timed = await browser.newPage();
    await timed.clock.install();
    let releaseRequest;
    let stall = false;
    await timed.route("**/api/**", async (route) => {
      if (route.request().method() === "POST") return route.fulfill({ status: 202, json: { image_id: imageId } });
      if (stall) await new Promise((resolve) => { releaseRequest = resolve; });
      await route.fulfill({ status: 404 }).catch(() => {}); // Request may have been aborted by its timeout.
    });
    await timed.goto(baseURL);
    await timed.locator("#file").setInputFiles({ name: "dog.png", mimeType: "image/png", buffer: png });
    await Promise.all([
      timed.waitForResponse((response) => response.status() === 404),
      timed.locator("#submit").click(),
    ]);
    await timed.clock.fastForward(40 * 60 * 1000 + 1);
    await timed.locator("#retry-result").waitFor({ state: "visible" });
    assert.match(await timed.locator("#status").textContent(), /Ожидание завершено/);
    stall = true;
    await Promise.all([
      timed.waitForRequest((request) => request.url().endsWith(`/api/images/${imageId}`)),
      timed.locator("#retry-result").click(),
    ]);
    await timed.clock.fastForward(15001);
    await timed.locator("#retry-result").waitFor({ state: "visible" });
    assert.match(await timed.locator("#status").textContent(), /Не удалось загрузить/);
    releaseRequest();
    await timed.close();
    const fallback = await browser.newPage();
    await fallback.addInitScript(() => { delete window.showDirectoryPicker; });
    await fallback.goto(baseURL);
    assert(await fallback.locator("#choose-folder").isHidden());
    assert.match(await fallback.locator("#folder").inputValue(), /ZIP/);
    const docs = await browser.newPage();
    const docsErrors = [];
    docs.on("pageerror", (error) => docsErrors.push(error.message));
    await docs.route("https://cdn.jsdelivr.net/**", (route) => route.fulfill({
      contentType: route.request().url().endsWith(".css") ? "text/css" : "application/javascript",
      body: route.request().url().endsWith(".css") ? "" : "window.SwaggerUIBundle = (config) => { window.swaggerConfig = config; document.getElementById('swagger-ui').textContent = config.spec.info.title; };",
    }));
    await docs.route("**/api/openapi.json", (route) => route.fulfill({ json: { openapi: "3.1.0", info: { title: "Test Image API", version: "1" }, paths: {} } }));
    await docs.goto(`${baseURL}/docs`);
    await docs.waitForFunction(() => window.swaggerConfig);
    assert.equal(new URL(docs.url()).pathname, "/api/docs");
    assert.deepEqual(await docs.evaluate(() => window.swaggerConfig.spec.servers), [{ url: "/api" }]);
    assert.deepEqual(docsErrors, []);
    console.log("Browser checks passed: automatic preview, compatible/blank refinements, literal caption, pending/error/retry/empty results, wait/request timeouts, polling stops on success/error/reset, automatic folder copy, permission/write failure fallback, ZIP, compact laptop/mobile layout, contained long-table scrolling, external submit, Swagger proxy and CSP.");
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
