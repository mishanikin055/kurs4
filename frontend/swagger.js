async function showDocs() {
  try {
    const response = await fetch("/api/openapi.json");
    if (!response.ok) throw new Error("schema_unavailable");
    const spec = await response.json();
    // The browser calls the API through nginx, including Swagger's Try it out.
    spec.servers = [{ url: "/api" }];
    window.ui = SwaggerUIBundle({ spec, dom_id: "#swagger-ui", deepLinking: true });
  } catch {
    document.getElementById("swagger-ui").textContent = "Не удалось загрузить документацию. Проверьте подключение к серверу и cdn.jsdelivr.net, затем обновите страницу.";
  }
}

showDocs();
