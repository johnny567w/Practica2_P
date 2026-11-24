// Interfaz: envía la imagen al servicio (POST /api/procesar) y muestra el resultado y los tiempos.
const $ = (id) => document.getElementById(id);
// Imagen transparente de 1x1: deja el recuadro vacío sin el ícono de «imagen rota».
const VACIA = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
let procesando = false;

const ETAPAS = [
  ["cliente_ms", "Total en el navegador", "desde que se envía la solicitud hasta recibir la imagen resultado (incluye la red)"],
  ["servidor_ms", "Servidor", "desde que llega la solicitud hasta tener la respuesta lista (incluye la espera en cola)"],
  ["espera_cola_ms", "Espera en cola", "tiempo esperando a que la GPU termine otras solicitudes"],
  ["decodificacion_ms", "Decodificación", "validar el archivo, decodificarlo y pasarlo a escala de grises"],
  ["preparacion_ms", "Preparación", "crear la máscara del filtro"],
  ["reserva_ms", "Reserva en GPU", "reservar memoria en la GPU (0 si se reutiliza la del mismo tamaño)"],
  ["h2d_ms", "CPU → GPU", "copiar la imagen y la máscara a la GPU (eventos CUDA, memoria paginable)"],
  ["kernel_ms", "Kernel", "la convolución en la GPU (eventos CUDA)"],
  ["d2h_ms", "GPU → CPU", "traer el resultado de la GPU (eventos CUDA)"],
  ["salida_ms", "Preparar la salida", "pasar a 8 bits y codificar en PNG"],
  ["escritura_ms", "Escritura", "guardar el PNG en el servidor"],
];

function fmt(ms) {
  if (ms === undefined || ms === null) return "–";
  return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms.toFixed(ms >= 10 ? 1 : 3) + " ms";
}

function mensaje(texto, tipo) {
  $("mensaje").textContent = texto;
  $("mensaje").className = "mensaje " + (tipo || "");
}

async function estado() {
  try {
    const r = await fetch("/api/salud");
    const d = await r.json();
    if (d.ok) {
      $("estado").textContent = `GPU lista: ${d.gpu} · inicialización ${fmt(d.inicializacion.total_ms)} · cola máxima ${d.cola_max}`;
      $("estado").className = "estado ok";
    } else {
      $("estado").textContent = "GPU no disponible: " + d.error;
      $("estado").className = "estado error";
    }
  } catch (e) {
    $("estado").textContent = "No se pudo consultar el servicio.";
    $("estado").className = "estado error";
  }
}

$("filtro").addEventListener("change", () => {
  $("grupo-k").style.display = $("filtro").value === "media" ? "" : "none";
});

$("imagen").addEventListener("change", () => {
  const f = $("imagen").files[0];
  if (f) $("vista-original").alt = "Imagen original";
  if (f) $("vista-original").src = URL.createObjectURL(f);
  $("vista-resultado").src = VACIA;                // el resultado anterior ya no corresponde
  $("descargar").classList.add("oculto");
});

$("formulario").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (procesando) return;                       // evita solicitudes duplicadas mientras se procesa
  const f = $("imagen").files[0];
  if (!f) { mensaje("Elija una imagen.", "error"); return; }
  procesando = true;
  $("boton").disabled = true;
  $("boton").textContent = "Procesando…";
  mensaje("Enviando y procesando en la GPU…");
  const datos = new FormData();
  datos.append("imagen", f);
  datos.append("filtro", $("filtro").value);
  datos.append("k", $("filtro").value === "media" ? $("k").value : "3");
  const t0 = performance.now();
  try {
    const r = await fetch("/api/procesar", { method: "POST", body: datos });
    const d = await r.json().catch(() => ({ ok: false, error: `Respuesta no válida (HTTP ${r.status})` }));
    if (!r.ok || !d.ok) throw new Error(d.error || `Error HTTP ${r.status}`);
    const img = await fetch(d.archivos.resultado);           // el tiempo total incluye descargar el resultado
    const blob = await img.blob();
    const total = performance.now() - t0;
    $("vista-resultado").src = URL.createObjectURL(blob);
    $("vista-resultado").alt = `Resultado: ${d.filtro} ${d.k}×${d.k}`;
    $("descargar").href = d.archivos.resultado + "?descargar=1";
    $("descargar").classList.remove("oculto");
    mostrarTiempos(d, total);
    mensaje(`Listo · solicitud ${d.id.slice(0, 8)}`, "ok");
  } catch (e) {
    mensaje(e.message, "error");
  } finally {
    procesando = false;
    $("boton").disabled = false;
    $("boton").textContent = "Procesar";
  }
});

function mostrarTiempos(d, total) {
  const t = { ...d.tiempos, cliente_ms: total };
  $("resumen").textContent = `${d.ancho}×${d.alto} píxeles · ${d.filtro} ${d.k}×${d.k} · ` +
    `${d.hilos_por_bloque} hilos por bloque · ${d.bloques.toLocaleString("es")} bloques · formato ${d.formato_entrada}`;
  $("tabla").innerHTML = "";
  for (const [clave, nombre, alcance] of ETAPAS) {
    const fila = document.createElement("tr");
    fila.innerHTML = `<td>${nombre}</td><td class="num">${fmt(t[clave])}</td><td>${alcance}</td>`;
    $("tabla").appendChild(fila);
  }
  $("tiempos").classList.remove("oculto");
}

estado();
