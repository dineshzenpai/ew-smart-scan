const form = document.querySelector("#simulation-form");
const button = form.querySelector("button");
const status = document.querySelector("#status");
const title = document.querySelector("#results-title");
const note = document.querySelector("#result-note");
const grid = document.querySelector("#metric-grid");
const latency = document.querySelector("#latency");

for (const input of form.querySelectorAll('input[type="range"]')) {
  const output = document.querySelector(`#${input.name}-value`);
  input.addEventListener("input", () => { output.value = input.value; });
}

function number(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function metric(label, value) {
  return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const parameters = new URLSearchParams(new FormData(form));
  button.disabled = true;
  status.textContent = "Running";
  status.className = "status running";
  note.textContent = "Running seeded episodes on the hosted Python simulation…";
  grid.hidden = true;
  latency.hidden = true;
  try {
    const response = await fetch(`/api/simulate?${parameters}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed");
    const data = payload.metrics;
    title.textContent = payload.method.replaceAll("_", " ");
    status.textContent = "Complete";
    status.className = "status ready";
    note.textContent = payload.note;
    grid.innerHTML = [
      metric("Probability of detection", number(data.Pd)),
      metric("False-alarm probability", number(data.Pfa)),
      metric("Intercepts / slot", number(data.intercept_rate)),
      metric("Average reward", number(data.average_reward)),
    ].join("");
    grid.hidden = false;
    const entries = Object.entries(payload.time_to_first_intercept).map(([kind, value]) => `${kind}: ${number(value, 1)} slots`);
    latency.textContent = entries.length ? `Average first-intercept latency — ${entries.join(" · ")}` : "No first-intercept latency available for this scenario.";
    latency.hidden = false;
  } catch (error) {
    status.textContent = "Could not run";
    status.className = "status";
    note.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
