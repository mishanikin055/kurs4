const $ = (id) => document.getElementById(id);
let folderHandle;
let imageId;

if (!("showDirectoryPicker" in window)) {
  $("choose-folder").hidden = true;
  $("folder").value = "В этом браузере доступна загрузка ZIP";
}

$("choose-folder").addEventListener("click", async () => {
  try {
    folderHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    $("folder").value = folderHandle.name;
  } catch (error) {
    if (error.name !== "AbortError") $("status").textContent = "Папка недоступна. Можно скачать ZIP.";
  }
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file || $("submit").disabled) return;
  $("submit").disabled = true;
  $("save-actions").hidden = true;
  imageId = null;
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
    $("status").textContent = `Отправлено в очередь. Папка в volume: ${accepted.output_folder}`;
    $("save-actions").hidden = false;
  } catch {
    $("status").textContent = "Не удалось подтвердить отправку. Проверьте подключение к серверу.";
  } finally {
    $("submit").disabled = false;
  }
});

async function resultFile(id, name) {
  const response = await fetch(`/api/images/${id}/files/${name}`);
  if (!response.ok) throw new Error("result_unavailable");
  return response;
}

$("save-copy").addEventListener("click", async () => {
  if (!imageId) return;
  const id = imageId;
  const selectedFolder = folderHandle;
  $("save-copy").disabled = true;
  $("submit").disabled = true;
  try {
    if (selectedFolder) {
      if (await selectedFolder.requestPermission({ mode: "readwrite" }) !== "granted") return;
      const manifest = await (await resultFile(id, "results.json")).blob();
      const directory = await selectedFolder.getDirectoryHandle(id, { create: true });
      for (const name of ["original", "normalized.png", "detection.png"]) {
        const response = await resultFile(id, name);
        const file = await directory.getFileHandle(name, { create: true });
        await response.body.pipeTo(await file.createWritable());
      }
      const file = await directory.getFileHandle("results.json", { create: true });
      const stream = await file.createWritable();
      await stream.write(manifest);
      await stream.close();
      $("status").textContent = `Копия сохранена: ${selectedFolder.name}/${id}. Основная копия остаётся в volume.`;
    } else {
      const response = await fetch(`/api/images/${id}/archive`);
      if (!response.ok) throw new Error("result_unavailable");
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `${id}.zip`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      $("status").textContent = "ZIP передан браузеру для сохранения. Основная копия остаётся в volume.";
    }
  } catch {
    $("status").textContent = "Копия не сохранена: файлы ещё не готовы или папка недоступна. Повторите сохранение позже.";
  } finally {
    $("save-copy").disabled = false;
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
