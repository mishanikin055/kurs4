const $ = (id) => document.getElementById(id);
let folderHandle;
let imageId;
let readyImageId;
let previewUrl;
let resultController;
let resultTimer;
let selectedObject;

function fitLabels() {
  if (!$("image-stage").clientWidth) return;
  for (const box of $("object-overlays").children) {
    const outline = box.querySelector(".object-outline");
    const marker = box.querySelector(".object-marker");
    marker.style.fontSize = "20px";
    const scale = Math.min(1, (outline.clientWidth - 4) / marker.offsetWidth, (outline.clientHeight - 2) / marker.offsetHeight);
    marker.style.fontSize = `${Math.max(1, 20 * scale)}px`;
  }
}

new ResizeObserver(fitLabels).observe($("image-stage"));

function selectObject(index, revealRow = false) {
  selectedObject = selectedObject === index ? null : index;
  const rows = [...$("objects").children];
  for (const container of [$("objects"), $("object-overlays")]) {
    [...container.children].forEach((element, i) => {
      const selected = i === selectedObject;
      element.classList.toggle("is-selected", selected);
      element.querySelector("button")?.setAttribute("aria-pressed", String(selected));
    });
  }
  const row = rows[selectedObject];
  $("selection-status").textContent = row
    ? `Выбран объект №${selectedObject + 1}: ${[...row.cells].slice(1).map((cell) => cell.textContent).filter(Boolean).join(", ")}.`
    : "Выделение снято.";
  if (revealRow && row) row.scrollIntoView({ block: "nearest" });
  fitLabels();
}

function renderObjects(result) {
  selectedObject = null;
  $("selection-status").textContent = "";
  $("objects").replaceChildren();
  $("object-overlays").replaceChildren();
  $("image-stage").style.setProperty("--image-ratio", result.width / result.height);
  $("image-stage").style.aspectRatio = `${result.width} / ${result.height}`;
  const captions = new Map(result.classification.objects.map((object) => [object.object_id, object.object_caption]));
  result.detection.objects.forEach((object, index) => {
    const row = document.createElement("tr");
    const number = document.createElement("button");
    number.type = "button";
    number.className = "object-number";
    number.textContent = index + 1;
    number.setAttribute("aria-label", `Выделить объект №${index + 1}: ${object.label}`);
    number.setAttribute("aria-pressed", "false");
    const cell = document.createElement("td");
    cell.append(number);
    row.append(cell);
    for (const value of [object.label, captions.get(object.object_id) ?? ""]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    row.addEventListener("click", () => selectObject(index));
    $("objects").append(row);

    const [x1, y1, x2, y2] = object.bbox;
    const box = document.createElement("div");
    box.className = "object-box";
    const outline = document.createElement("span");
    outline.className = "object-outline";
    outline.style.left = `${x1 / result.width * 100}%`;
    outline.style.top = `${y1 / result.height * 100}%`;
    outline.style.width = `${(x2 - x1) / result.width * 100}%`;
    outline.style.height = `${(y2 - y1) / result.height * 100}%`;
    const marker = document.createElement("span");
    marker.className = "object-marker";
    marker.textContent = object.label;
    marker.setAttribute("aria-hidden", "true");
    outline.append(marker);
    box.append(outline);
    $("object-overlays").append(box);
  });
  $("image-stage").onclick = (event) => {
    const image = $("image-stage").getBoundingClientRect();
    const x = (event.clientX - image.left) / image.width * result.width;
    const y = (event.clientY - image.top) / image.height * result.height;
    let hit = null;
    let smallestArea = Infinity;
    result.detection.objects.forEach((object, index) => {
      const [x1, y1, x2, y2] = object.bbox;
      const area = (x2 - x1) * (y2 - y1);
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && area < smallestArea) {
        hit = index;
        smallestArea = area;
      }
    });
    selectObject(hit, true);
  };
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") selectObject(null);
});

function stopWaiting() {
  clearTimeout(resultTimer);
  resultController?.abort();
}

function showResultError(message) {
  $("status").textContent = message;
  $("preview-placeholder").textContent = "Результат пока недоступен.";
  $("retry-result").hidden = false;
}

if (!("showDirectoryPicker" in window)) {
  $("choose-folder").hidden = true;
  $("folder").value = "В этом браузере доступна загрузка ZIP";
}

$("choose-folder").addEventListener("click", async () => {
  try {
    folderHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    $("folder").value = folderHandle.name;
    if (readyImageId && readyImageId === imageId) await saveToFolder(imageId, folderHandle);
  } catch (error) {
    if (error.name !== "AbortError") $("status").textContent = "Папка недоступна. Можно скачать ZIP.";
  }
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file || $("submit").disabled) return;
  stopWaiting();
  $("retry-result").hidden = true;
  $("submit").disabled = true;
  $("save-actions").hidden = true;
  $("result").hidden = true;
  $("result-placeholder").hidden = false;
  $("result-preview").hidden = true;
  $("caption").hidden = true;
  $("caption").textContent = "";
  $("annotation-placeholder").hidden = false;
  $("preview-placeholder").hidden = false;
  $("preview-placeholder").textContent = "После обработки здесь появится изображение с рамками.";
  $("result-image").removeAttribute("src");
  $("object-overlays").replaceChildren();
  $("selection-status").textContent = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imageId = null;
  readyImageId = null;
  $("status").textContent = "Отправка изображения…";
  const data = new FormData();
  data.append("file", file);
  try {
    const response = await fetch("/api/images", { method: "POST", body: data });
    if (!response.ok) {
      $("status").textContent = "Отправка не подтверждена. Проверьте файл и доступность сервера.";
      return;
    }
    const accepted = await response.json();
    imageId = accepted.image_id;
    $("status").textContent = "Изображение обрабатывается. Результат появится автоматически…";
    $("preview-placeholder").textContent = "Ожидаем результат обработки…";
    waitForResult(imageId);
  } catch {
    $("status").textContent = "Не удалось подтвердить отправку. Проверьте подключение к серверу.";
  } finally {
    $("submit").disabled = false;
  }
});

async function resultFile(id, name) {
  const response = await fetch(`/api/images/${id}/files/${name}`);
  if (!response.ok) throw new Error(response.status === 404 ? "result_pending" : "result_unavailable");
  return response;
}

async function waitForResult(id, deadline = Date.now() + 40 * 60 * 1000) {
  const controller = new AbortController();
  resultController = controller;
  $("retry-result").hidden = true;
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`/api/images/${id}`, { signal: controller.signal, cache: "no-store" });
    if (response.status === 404) {
      if (Date.now() >= deadline) {
        showResultError("Ожидание завершено. Результат пока не получен. Можно повторить ожидание.");
      } else {
        resultTimer = setTimeout(() => waitForResult(id, deadline), 2000);
      }
      return;
    }
    if (!response.ok) throw new Error("result_unavailable");
    const result = await response.json();
    const imageResponse = await fetch(`/api/images/${id}/files/normalized.png`, { signal: controller.signal });
    if (!imageResponse.ok) throw new Error("result_unavailable");
    const image = await imageResponse.blob();
    if (id !== imageId) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(image);
    $("result-image").src = previewUrl;
    $("result-image").width = result.width;
    $("result-image").height = result.height;
    $("caption").textContent = result.annotation.raw_caption;
    $("caption").hidden = false;
    $("annotation-placeholder").hidden = true;
    renderObjects(result);
    $("no-objects").hidden = result.detection.objects.length > 0;
    $("objects-table").hidden = result.detection.objects.length === 0;
    $("result").hidden = false;
    $("result-placeholder").hidden = true;
    $("result-preview").hidden = false;
    $("preview-placeholder").hidden = true;
    fitLabels();
    readyImageId = id;
    $("save-actions").hidden = Boolean(folderHandle);
    $("status").textContent = "Результат готов.";
    clearTimeout(timeout);
    if (folderHandle) await saveToFolder(id, folderHandle);
  } catch {
    if (id === imageId && resultController === controller) {
      showResultError("Не удалось загрузить результат. Проверьте подключение и повторите ожидание.");
    }
  } finally {
    clearTimeout(timeout);
  }
}

$("retry-result").addEventListener("click", () => {
  if (!imageId) return;
  stopWaiting();
  $("status").textContent = "Ожидаем результат. Он появится автоматически…";
  $("preview-placeholder").textContent = "Ожидаем результат обработки…";
  waitForResult(imageId);
});

async function saveToFolder(id, selectedFolder) {
  $("submit").disabled = true;
  $("choose-folder").disabled = true;
  $("save-actions").hidden = true;
  $("status").textContent = "Результат готов. Сохраняем файлы в выбранную папку…";
  try {
    // Automatic saving cannot request browser permission without a user gesture.
    if (await selectedFolder.queryPermission({ mode: "readwrite" }) !== "granted") {
      throw new Error("folder_permission_required");
    }
    const manifest = await (await resultFile(id, "results.json")).blob();
    const result = JSON.parse(await manifest.text());
    const directory = await selectedFolder.getDirectoryHandle(id, { create: true });
    for (const name of ["original", "normalized.png", "detection.png"]) {
      const response = await resultFile(id, name);
      const file = await directory.getFileHandle(name, { create: true });
      await response.body.pipeTo(await file.createWritable());
    }
    if (result.crops) {
      const crops = await directory.getDirectoryHandle("crops", { create: true });
      for (const name of result.crops) {
        const response = await resultFile(id, `crops/${name}`);
        const file = await crops.getFileHandle(name, { create: true });
        await response.body.pipeTo(await file.createWritable());
      }
    }
    const file = await directory.getFileHandle("results.json", { create: true });
    const stream = await file.createWritable();
    await stream.write(manifest);
    await stream.close();
    $("status").textContent = `Результат готов. Копия сохранена в папке «${selectedFolder.name}».`;
  } catch {
    $("status").textContent = "Результат готов, но сохранить полный комплект не удалось. Выберите папку снова или скачайте ZIP.";
    $("save-actions").hidden = false;
  } finally {
    $("choose-folder").disabled = false;
    $("submit").disabled = false;
  }
}

$("download-zip").addEventListener("click", async () => {
  if (!readyImageId || readyImageId !== imageId) return;
  $("download-zip").disabled = true;
  $("choose-folder").disabled = true;
  $("submit").disabled = true;
  try {
    const response = await fetch(`/api/images/${imageId}/archive`);
    if (!response.ok) throw new Error("result_unavailable");
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `${imageId}.zip`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    $("status").textContent = "ZIP передан браузеру для сохранения.";
  } catch {
    $("status").textContent = "Не удалось скачать ZIP. Проверьте подключение и повторите загрузку ZIP.";
  } finally {
    $("download-zip").disabled = false;
    $("choose-folder").disabled = false;
    $("submit").disabled = false;
  }
});

$("dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("dropzone").classList.add("dragging");
});
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
$("dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("dragging");
  if (!event.dataTransfer.files.length) return;
  const transfer = new DataTransfer();
  transfer.items.add(event.dataTransfer.files[0]);
  $("file").files = transfer.files;
});
