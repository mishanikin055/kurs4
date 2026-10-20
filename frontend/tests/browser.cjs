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
        getDirectoryHandle: async (directory) => makeDirectory(directory),
      });
      function makeDirectory(directory) {
        return {
          getDirectoryHandle: async (name) => makeDirectory(`${directory}/${name}`),
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
        };
      }
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const requests = [];
    let ready = false;
    let uploadError = false;
    let resultError = false;
    const result = {
      id: imageId, width: 640, height: 480,
      crops: ["dog.png", "cat.png"],
      detection: { objects: [{ object_id: "dog", label: "dog", bbox: [32, 24, 240, 432] }, { object_id: "cat", label: "cat", bbox: [40, 30, 640, 480] }] },
      classification: { objects: [{ object_id: "cat", label: "unicycle", refinement: null }, { object_id: "dog", label: "Labrador retriever", refinement: "Labrador retriever" }] },
      annotation: { raw_caption: "  A dog and <b>a cat</b>.  " },
    };
    let preview = result;
    const files = {
      original: png,
      "crops/dog.png": png,
      "crops/cat.png": png,
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
      const name = path.split("/files/")[1];
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
    assert.equal(await page.locator(".page-header, .brand, #object-help").count(), 0, "Project header and selection hint are removed");
    assert(await page.locator("#caption").evaluate((caption) => Boolean(caption.closest(".annotation-panel"))), "Caption has its own annotation panel");
    assert.equal(await page.locator(".result-panel #caption").count(), 0, "Object results no longer contain the caption");
    assert.equal(await page.locator("#annotation-title").textContent(), "Результат аннотирования");
    assert(await page.locator("#annotation-placeholder").isHidden());
    assert(await page.locator("#caption").evaluate((caption) => {
      const panel = caption.closest(".annotation-panel");
      const style = getComputedStyle(panel);
      const bottomSpace = panel.getBoundingClientRect().bottom - caption.getBoundingClientRect().bottom;
      return bottomSpace <= parseFloat(style.paddingBottom) + parseFloat(style.borderBottomWidth) + 1;
    }), "Short annotation panel ends after the text and normal padding");
    assert.deepEqual(await page.locator("#objects tr").allTextContents(), ["1dogLabrador retriever", "2cat"]);
    assert.deepEqual(await page.locator(".object-marker").allTextContents(), ["dog", "cat"], "Image labels contain classes without object numbers");
    assert.equal(await page.locator("#object-overlays button").count(), 0, "Object numbers are clickable only in the table");
    assert.equal(await page.locator(".object-box:visible").count(), 2, "All boxes are shown by default");
    assert.equal(await page.locator("[aria-pressed=true]").count(), 0);
    assert.deepEqual(await page.locator(".object-outline").first().evaluate((box) => [getComputedStyle(box).borderColor, getComputedStyle(box).borderWidth]), ["rgb(255, 0, 0)", "2px"], "Default frames are bright red and more visible");
    assert(requests.includes(`/api/images/${imageId}/files/normalized.png`), "Old and new results use clean source pixels with the same numbered overlay");
    await page.locator("#objects tr").nth(1).locator("td").nth(1).click();
    assert(await page.locator(".object-box").nth(1).evaluate((box) => box.classList.contains("is-selected")), "Clicking a row selects its box");
    assert.equal(await page.locator(".object-box:visible").count(), 2, "Selection keeps all boxes and class names visible");
    assert.deepEqual(await page.locator(".object-outline").nth(1).evaluate((box) => [getComputedStyle(box).borderColor, getComputedStyle(box).borderWidth]), ["rgb(7, 89, 133)", "2px"], "Selected frame turns blue");
    await page.locator(".object-number").first().focus();
    await page.keyboard.press("Enter");
    assert(await page.locator(".object-box").first().evaluate((box) => box.classList.contains("is-selected")), "Object selection works from the keyboard");
    assert.equal(await page.locator(".object-box.is-selected").count(), 1, "Only one object is selected");
    await page.keyboard.press("Escape");
    assert.equal(await page.locator(".object-box.is-selected").count(), 0, "Escape clears the frame highlight");
    await page.locator(".object-number").nth(1).click();
    assert.equal(await page.locator(".object-number").nth(1).getAttribute("aria-pressed"), "true", "Clicking a table number selects its frame");
    assert.equal(await page.locator("#objects tr.is-selected").count(), 1);
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("[aria-pressed=true]").count(), 0, "Escape clears both selections");
    const stage = await page.locator("#image-stage").boundingBox();
    await page.locator("#image-stage").click({ position: { x: stage.width * 100 / result.width, y: stage.height * 100 / result.height } });
    assert.equal(await page.locator(".object-number").first().getAttribute("aria-pressed"), "true", "Clicking overlapping objects selects the smallest containing box and matching row");
    await page.locator("#image-stage").click({ position: { x: stage.width * 600 / result.width, y: stage.height * 450 / result.height } });
    assert.equal(await page.locator(".object-number").nth(1).getAttribute("aria-pressed"), "true", "Clicking the second object updates the selected row");
    await page.keyboard.press("Escape");
    assert.equal(await page.locator(".object-box.is-selected").count(), 0, "Escape also works after image clicks");
    await page.locator("#image-stage").click({ position: { x: stage.width * 100 / result.width, y: stage.height * 100 / result.height } });
    await page.locator("#image-stage").click({ position: { x: 2, y: 2 } });
    assert.equal(await page.locator(".object-box.is-selected").count(), 0, "Clicking outside all objects clears the selection");
    const controls = await page.locator(".controls").boundingBox();
    const image = await page.locator("#result-image").boundingBox();
    assert(image.x >= controls.x + controls.width, "Image is to the right of the controls and table");
    assert(await page.locator("#submit").evaluate((button) => button.form.id === "upload-form" && !button.closest("#upload-form") && !button.closest("#dropzone")), "Submit is outside both upload blocks and still submits the form");
    assert(await page.locator("#result").evaluate((result) => Boolean(result.closest(".controls"))), "Results are below the form in the left column");
    for (const viewport of [{ width: 1366, height: 650 }, { width: 1280, height: 600 }, { width: 1536, height: 730 }, { width: 1024, height: 600 }]) {
      await page.setViewportSize(viewport);
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight && document.documentElement.scrollWidth <= innerWidth), `No page scrolling at ${viewport.width}x${viewport.height}`);
      for (const selector of ["#submit", ".result-panel", ".preview", ".annotation-panel"]) {
        const box = await page.locator(selector).boundingBox();
        assert(box.y >= 0 && box.y + box.height <= viewport.height, `${selector} fits the laptop viewport`);
      }
      const picture = await page.locator("#result-image").boundingBox();
      const title = await page.locator("#preview-title").boundingBox();
      const frame = await page.locator(".preview").boundingBox();
      const table = await page.locator("#objects-table").boundingBox();
      const results = await page.locator(".result-panel").boundingBox();
      const annotation = await page.locator(".annotation-panel").boundingBox();
      const upload = await page.locator("#upload-form").boundingBox();
      assert(upload.y <= 20 && frame.y <= 20, "Controls and image move up into the former header space");
      assert(picture.y - (title.y + title.height) <= 20, "Image starts directly below its title");
      assert(frame.y + frame.height - (picture.y + picture.height) <= 20, "Image frame ends at the image");
      assert(annotation.y >= frame.y + frame.height && annotation.y - (frame.y + frame.height) <= 20, "Annotation panel is directly below the image");
      assert(Math.abs(annotation.x - frame.x) < 1 && Math.abs(annotation.width - frame.width) < 1, "Annotation panel is aligned with the image");
      const caption = await page.locator("#caption").boundingBox();
      assert(annotation.y + annotation.height - (caption.y + caption.height) <= 20, "Annotation frame ends at the text");
      assert(results.y + results.height - (table.y + table.height) <= 16, "Object frame ends at the table");
      for (let index = 0; index < result.detection.objects.length; index++) {
        const [x1, y1, x2, y2] = result.detection.objects[index].bbox;
        const box = await page.locator(".object-outline").nth(index).boundingBox();
        assert(Math.abs(box.x - (picture.x + x1 / result.width * picture.width)) < 1, "Box stays aligned horizontally after resizing");
        assert(Math.abs(box.y - (picture.y + y1 / result.height * picture.height)) < 1, "Box stays aligned vertically after resizing");
        assert(Math.abs(box.width - (x2 - x1) / result.width * picture.width) < 1);
        assert(Math.abs(box.height - (y2 - y1) / result.height * picture.height) < 1);
        const marker = await page.locator(".object-marker").nth(index).boundingBox();
        assert(marker.x >= picture.x - 1 && marker.y >= picture.y - 1 && marker.x + marker.width <= picture.x + picture.width + 1 && marker.y + marker.height <= picture.y + picture.height + 1, "Edge labels stay inside the image");
        assert(marker.x >= box.x && marker.y >= box.y && marker.x + marker.width <= box.x + box.width + 1 && marker.y + marker.height <= box.y + box.height + 1, "Class text fits inside its own frame");
        assert.equal(await page.locator(".object-marker").nth(index).evaluate((label) => getComputedStyle(label).backgroundColor), "rgba(0, 0, 0, 0)", "Class text has no background");
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
    await page.locator("#caption").evaluate((caption) => { caption.textContent = "Long annotation. ".repeat(500); });
    assert(await page.locator("#caption").evaluate((caption) => caption.scrollHeight > caption.clientHeight), "Long annotations scroll inside their own panel");
    assert(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight), "Long annotation does not make the laptop page scroll");
    await page.locator("#caption").evaluate((caption, text) => { caption.textContent = text; }, result.annotation.raw_caption);
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
    assert.equal(await page.evaluate(() => Object.keys(window.savedFiles).length), 6, "Folder selected before upload receives all files and crops automatically");
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
    await page.locator(".object-number").nth(1).click();
    assert.equal(await page.locator(".object-number").nth(1).getAttribute("aria-pressed"), "true");
    preview = { ...result, detection: { objects: [] }, classification: { objects: [] } };
    await page.locator("#submit").click();
    await page.waitForFunction(() => !document.getElementById("no-objects").hidden);
    assert(await page.locator("#objects-table").isHidden());
    assert.equal(await page.locator(".object-marker").count(), 0, "New empty result clears all old boxes");
    assert.equal(await page.locator("#object-help").count(), 0);
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
    assert(await page.locator("#caption").isHidden(), "New upload clears the old annotation");
    assert.equal(await page.locator("#caption").textContent(), "");
    assert(await page.locator("#annotation-placeholder").isVisible());
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
    console.log("Browser checks passed: separate large annotation panel below image, removed header/hint, bright-red frames, long caption/table contained scrolling, caption reset on upload, class text inside boxes without backgrounds, numbers only in table, blue frame/row selection by image click/table/keyboard, smallest overlapping box selected, background click and Escape clear selection, box alignment on resize, automatic preview, compatible/blank refinements, literal caption, pending/error/retry/empty results, wait/request timeouts, polling stops on success/error/reset, automatic folder copy with crops, permission/write failure fallback, ZIP, compact laptop/mobile layout, external submit, Swagger proxy and CSP.");
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
