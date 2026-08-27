const form = document.querySelector("#simulation-form");
const button = form.querySelector("button");
const status = document.querySelector("#status");
const title = document.querySelector("#results-title");
const note = document.querySelector("#result-note");
const grid = document.querySelector("#metric-grid");
const latency = document.querySelector("#latency");
const resultMeta = document.querySelector("#result-meta");

for (const input of form.querySelectorAll('input[type="range"]')) {
  const output = document.querySelector(`#${input.name}-value`);
  input.addEventListener("input", () => { output.value = input.value; });
}

function number(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function percent(value, digits = 1) {
  return value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
}

function metric(label, value, detail, primary = false) {
  return `<article class="metric${primary ? " primary" : ""}"><span>${label}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const parameters = new URLSearchParams(new FormData(form));
  button.disabled = true;
  status.textContent = "Scanning";
  status.className = "status running";
  title.textContent = "Scanning the RF environment…";
  note.textContent = "Learning from noisy single-band observations. This can take a few seconds.";
  grid.hidden = true;
  latency.hidden = true;
  resultMeta.textContent = "COMPUTING SEEDED TRIALS";

  try {
    const response = await fetch(`/api/simulate?${parameters}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Simulation request failed");

    const data = payload.metrics;
    const lockedEmitters = `${Math.round(data.detected_emitters)}/${Math.round(data.total_emitters)}`;
    title.textContent = payload.method.replaceAll("_", " ");
    status.textContent = "Complete";
    status.className = "status ready";
    note.textContent = payload.note;
    grid.innerHTML = [
      metric("Detection confidence", percent(data.Pd), "Primary intercept objective", true),
      metric("False alarms", percent(data.Pfa), "Sensor noise penalty"),
      metric("Intercept cadence", number(data.intercept_rate), "Confirmed looks / slot"),
      metric("Prediction accuracy", percent(data.prediction_accuracy), "Next-band estimate"),
      metric("Emitter lock", lockedEmitters, "Threats detected"),
    ].join("");
    grid.hidden = false;

    const entries = Object.entries(payload.time_to_first_intercept)
      .map(([kind, value]) => `${kind}: ${number(value, 1)} slots`);
    latency.textContent = entries.length
      ? `FIRST INTERCEPT LATENCY — ${entries.join("  •  ")}`
      : "No first-intercept latency available for this scenario.";
    latency.hidden = false;
    resultMeta.textContent = `${payload.scenario.bands} BANDS · ${payload.scenario.episodes} SEEDED EPISODES · HORIZON ${payload.scenario.horizon}`;
  } catch (error) {
    status.textContent = "Unavailable";
    status.className = "status";
    title.textContent = "Scan request could not complete.";
    note.textContent = error.message;
    resultMeta.textContent = "VERIFY CONFIGURATION AND TRY AGAIN";
  } finally {
    button.disabled = false;
  }
});

