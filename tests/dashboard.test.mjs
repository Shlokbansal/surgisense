import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { chartPosition, formatPercent, formatScore, validateSummary } from "../docs/app.mjs";

const summary = JSON.parse(readFileSync(new URL("../docs/results.json", import.meta.url), "utf8"));

test("public dashboard data has the expected shape and no patient-level fields", () => {
  assert.equal(validateSummary(summary), summary);
  const serialized = JSON.stringify(summary);
  assert.doesNotMatch(serialized, /patient_ids|sample_id|gene_ids|duration_months/);
  assert.equal(summary.cohort.eligible_patients, 501);
  assert.equal(summary.rna_cohort.matched_patients, 497);
  assert.match(summary.datahub_commit, /^[0-9a-f]{40}$/);
});

test("chart values and labels are stable at boundaries", () => {
  assert.equal(chartPosition(0.3, 0.3, 0.8), 0);
  assert.equal(chartPosition(0.5, 0.3, 0.8), 40);
  assert.equal(chartPosition(0.8, 0.3, 0.8), 100);
  assert.equal(chartPosition(1, 0.3, 0.8), 100);
  assert.equal(formatScore(0.567327766), "0.567");
  assert.equal(formatPercent(0.189483), "18.9%");
});

test("summary validates cohort arithmetic", () => {
  const broken = structuredClone(summary);
  broken.cohort.censored_patients = 0;
  assert.throws(() => validateSummary(broken), /Cohort counts do not add up/);
});
