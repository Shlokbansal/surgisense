const CINDEX_MIN = 0.3;
const CINDEX_MAX = 0.8;
const BRIER_MAX = 0.25;

export function formatScore(value) {
  if (!Number.isFinite(value)) throw new Error("A metric is not finite");
  return value.toFixed(3);
}

export function formatPercent(value) {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("A probability is outside zero to one");
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function chartPosition(value, minimum, maximum) {
  if (!Number.isFinite(value) || maximum <= minimum) throw new Error("Invalid chart value");
  return Math.max(0, Math.min(100, ((value - minimum) / (maximum - minimum)) * 100));
}

export function validateSummary(summary) {
  if (summary.schema_version !== 1) throw new Error("Unsupported dashboard data version");
  if (!/^[0-9a-f]{40}$/.test(summary.datahub_commit)) {
    throw new Error("Source snapshot is invalid");
  }
  if (!summary.cohort || !summary.rna_cohort || !summary.experiments) {
    throw new Error("Dashboard data is incomplete");
  }
  if (summary.cohort.eligible_patients !==
      summary.cohort.observed_deaths + summary.cohort.censored_patients) {
    throw new Error("Cohort counts do not add up");
  }
  for (const key of ["clinical_tmb", "rna"]) {
    const experiment = summary.experiments[key];
    if (!experiment || experiment.models.length !== 2 || experiment.horizons.length !== 2) {
      throw new Error(`Experiment ${key} is incomplete`);
    }
    for (const model of experiment.models) {
      formatScore(model.held_out_c_index);
      if (model.held_out_c_index_interval.length !== 2 ||
          model.held_out_c_index_interval[0] > model.held_out_c_index_interval[1]) {
        throw new Error("C-index interval is invalid");
      }
    }
    for (const horizon of experiment.horizons) {
      formatPercent(horizon.observed_death_probability);
      if (horizon.models.length !== 2) throw new Error("Horizon results are incomplete");
    }
  }
  return summary;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderStats(summary) {
  const target = document.getElementById("cohort-stats");
  target.replaceChildren();
  const cards = [
    [summary.cohort.eligible_patients, "Eligible patients", "With usable survival follow-up"],
    [summary.cohort.observed_deaths, "Observed deaths", "Others are right-censored"],
    [summary.rna_cohort.matched_patients, "RNA-matched patients", `${summary.rna_cohort.excluded_without_rna} lack a matching RNA sample`],
    [summary.rna_cohort.retained_unique_genes, "Unambiguous genes", "Before train-only feature selection"],
  ];
  for (const [number, label, detail] of cards) {
    const card = element("div", "stat-card");
    card.append(element("span", "stat-number", number.toLocaleString()));
    card.append(element("span", "stat-label", label));
    card.append(element("span", "stat-sub", detail));
    target.append(card);
  }
  const missing = summary.cohort.missing_features;
  document.getElementById("missingness-note").textContent =
    `Missing predictor values: age ${missing.age}, sex ${missing.sex}, stage ${missing.stage}, ` +
    `TMB ${missing.tmb}. These were handled inside training folds, not before the patient split.`;
  const source = document.getElementById("source-version");
  source.replaceChildren(document.createTextNode("Pinned cBioPortal DataHub snapshot: "));
  const link = element("a", "", summary.datahub_commit.slice(0, 12));
  link.href = `https://github.com/cBioPortal/datahub/tree/${summary.datahub_commit}/public/luad_tcga_pan_can_atlas_2018`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  source.append(link, document.createTextNode(". Source files are verified by SHA-256 before analysis."));
}

function renderCIndex(experiment) {
  const target = document.getElementById("cindex-chart");
  target.replaceChildren();
  experiment.models.forEach((model, index) => {
    const row = element("div", `metric-row ${index === 1 ? "molecular" : "clinical"}`);
    row.append(element("span", "metric-label", model.label));
    const track = element("div", "ci-track");
    const [low, high] = model.held_out_c_index_interval;
    track.setAttribute(
      "aria-label",
      `${model.label}: C-index ${formatScore(model.held_out_c_index)}, 95% interval ${formatScore(low)} to ${formatScore(high)}`
    );
    track.append(element("div", "ci-base"));
    track.append(element("div", "ci-chance"));
    const interval = element("div", "ci-interval");
    const start = chartPosition(low, CINDEX_MIN, CINDEX_MAX);
    const end = chartPosition(high, CINDEX_MIN, CINDEX_MAX);
    interval.style.left = `${start}%`;
    interval.style.width = `${end - start}%`;
    track.append(interval);
    const dot = element("div", "ci-dot");
    dot.style.left = `${chartPosition(model.held_out_c_index, CINDEX_MIN, CINDEX_MAX)}%`;
    track.append(dot);
    row.append(track);
    row.append(element("span", "metric-value", formatScore(model.held_out_c_index)));
    target.append(row);
  });
  const axis = element("div", "chart-axis");
  for (const tick of [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) {
    axis.append(element("span", "", tick.toFixed(2)));
  }
  target.append(axis);
}

function renderBrier(experiment, months) {
  const horizon = experiment.horizons.find((item) => item.months === months);
  if (!horizon) throw new Error(`Missing ${months}-month result`);
  const target = document.getElementById("brier-chart");
  target.replaceChildren();
  const rows = [
    ["Constant reference", horizon.constant_reference_brier, "reference"],
    [experiment.models[0].label, horizon.models[0].brier, "clinical"],
    [experiment.models[1].label, horizon.models[1].brier, "molecular"],
  ];
  for (const [label, score, kind] of rows) {
    const row = element("div", `bar-row ${kind}`);
    row.append(element("span", "bar-label", label));
    const track = element("div", "bar-track");
    track.setAttribute("aria-label", `${label}: Brier score ${formatScore(score)}`);
    const fill = element("div", "bar-fill");
    fill.style.width = `${chartPosition(score, 0, BRIER_MAX)}%`;
    track.append(fill);
    row.append(track);
    row.append(element("span", "bar-value", formatScore(score)));
    target.append(row);
  }
  const detail = document.getElementById("probability-detail");
  detail.replaceChildren();
  const heading = element("strong", "", `${months === 12 ? "One" : "Two"}-year death probability: `);
  detail.append(heading);
  detail.append(document.createTextNode(
    `observed ${formatPercent(horizon.observed_death_probability)}; ` +
    `${experiment.models[0].label.toLowerCase()} predicted ${formatPercent(horizon.models[0].mean_predicted_death)}; ` +
    `${experiment.models[1].label.toLowerCase()} predicted ${formatPercent(horizon.models[1].mean_predicted_death)}. ` +
    `${horizon.held_out_deaths} deaths observed by this time among ${experiment.held_out_patients} held-out patients.`
  ));
}

function renderExperiment(summary, key, months) {
  const experiment = summary.experiments[key];
  const isRna = key === "rna";
  const population = isRna ? summary.rna_cohort.matched_patients : summary.cohort.eligible_patients;
  document.getElementById("experiment-summary").textContent =
    `${population} patients · ${experiment.development_patients} development · ` +
    `${experiment.held_out_patients} held out · ${experiment.held_out_deaths} held-out deaths. ` +
    (isRna ? `${summary.rna_cohort.excluded_without_rna} clinical patients lacked matching RNA and were excluded from both models.` :
      "Both models use the same clinical patient split.");
  renderCIndex(experiment);
  renderBrier(experiment, months);
  const interpretation = document.getElementById("result-interpretation");
  interpretation.replaceChildren();
  interpretation.append(element("strong", "", "What this means. "));
  interpretation.append(document.createTextNode(
    isRna ?
      "Adding RNA looked helpful during development but performed worse on the held-out patients. This can happen when a model learns patterns that do not carry over. It is not proof that RNA is uninformative in general." :
      "Adding TMB did not improve held-out ranking. The difference is small and the intervals are wide, so this does not establish whether TMB is or is not a useful biomarker elsewhere."
  ));
}

async function start() {
  try {
    const response = await fetch(new URL("./results.json", import.meta.url));
    if (!response.ok) throw new Error(`Could not load results (${response.status})`);
    const summary = validateSummary(await response.json());
    renderStats(summary);
    let experiment = "clinical_tmb";
    let months = 12;
    const refresh = () => renderExperiment(summary, experiment, months);
    document.querySelectorAll("[data-experiment]").forEach((button) => {
      button.addEventListener("click", () => {
        experiment = button.dataset.experiment;
        document.querySelectorAll("[data-experiment]").forEach((other) => {
          const active = other === button;
          other.classList.toggle("active", active);
          other.setAttribute("aria-pressed", String(active));
        });
        refresh();
      });
    });
    document.querySelectorAll("[data-months]").forEach((button) => {
      button.addEventListener("click", () => {
        months = Number(button.dataset.months);
        document.querySelectorAll("[data-months]").forEach((other) => {
          const active = other === button;
          other.classList.toggle("active", active);
          other.setAttribute("aria-pressed", String(active));
        });
        refresh();
      });
    });
    refresh();
  } catch (error) {
    document.getElementById("cohort-stats").textContent =
      "Research results could not be loaded. Please refresh or view the source repository.";
    document.getElementById("experiment-summary").textContent = error.message;
  }
}

if (typeof document !== "undefined") start();
