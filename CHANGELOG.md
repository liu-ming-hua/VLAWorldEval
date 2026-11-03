# Changelog

Versioned releases freeze framework code and configuration. Model weights, datasets and
third-party runtimes are obtained separately under their own terms.

## Unreleased

- Add native Ctrl-World recorded-state, hold-pose and shuffled-state controls through
  the shared branch executor; separate prediction sensitivity from reference accuracy.
- Include synchronized GT comparisons, temporal metrics and inclusive-tail boundary
  diagnostics in offline reports. Two released examples are engineering evidence,
  not a physical reliability certification.

- Add a read-only world-model route inventory (`python -m cli world-models`) that
  distinguishes registered adapters, replay configurations and acceptance evidence.
- Integrate A2World input preparation and GE-Sim finite-tail autoregressive replay
  into the shared executor, with managed server lifecycle and chunk-boundary diagnostics.
- Add optional τ₀-WM full-video transport, explicit RGB validation and generated-frame
  feedback. This is a project compatibility patch, not an upstream API guarantee.
- Document native-domain blockers: A2World requires a gated tokenizer; τ₀-WM's
  released simulator does not match the local TacoPlay sample. Protocol tests do
  not establish checkpoint quality or real-action replay validity.
- Keep runtime assets and intermediate experiment outputs outside source distributions.

## 0.1.0 — 2026-10-31

First versioned release of **VLAWorldEval**, a modular VLA × world-model evaluation framework.

### Included

- Independent policy and world-model adapters, explicit action/view contracts, isolated
  model environments and preflight checks for assets, inputs and GPU allocation.
- Autoregressive policy loops and recorded-action replay through the shared executor;
  continuous videos, action arrays, configuration and provenance saved together.
- Seeded action-branch comparisons, native-entry consistency checks and reference-aware
  reports that retain failed and unrun planned cases.
- Portable offline reports with synchronized views, plus a weight-free quick start that
  exercises 30 synthetic loop iterations and clearly labels its results.
- English and Chinese manuals, installation/download recipes and an explicit support matrix.

### Verified scope

- Local source validation: **1,201 weight-free tests**, fresh wheel/source installation,
  and strict bilingual documentation build. The release workflow repeats the engineering
  and installation checks on the tagged source.
- Existing real-weight acceptance covers OpenVLA + WorldGym and scoped native-entry
  parity for Ctrl-World and the community Cosmos DROID checkpoint. Five native long
  replays completed with 88 world-model calls. These are small engineering acceptance
  cases, not a benchmark ranking or evidence that every registered pair works.
- Scope and pinned evidence: `configs/native_parity_evidence_v1.json` and
  `configs/world_model_acceptance_v1.json`; see the manuals for individual pair support.

### Limitations

- World-model action following can be unreliable. Visual quality or reference similarity
  does not establish physical correctness, task success or reliable VLA ranking.
- Adapters still require compatible robot embodiment, action semantics, camera views,
  input state and model training domain. Registration alone is not verified compatibility.
- This release does not bundle pretrained weights, datasets or third-party model source.
  Downloads may require upstream credentials or acceptance of model/data terms.

### Get started

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/liu-ming-hua/VLAWorldEval.git
cd VLAWorldEval
bash scripts/setup/core.sh
environment/core/bin/python -m cli demo --output results/first_demo --chunks 30
environment/core/bin/python -m cli report \
  --source results/first_demo/summary.json --output results/first_report
```

Requires Python 3.11, uv and FFmpeg. Open `results/first_report/index.html` in a browser.
For real models, follow the [quick start](docs/en/quickstart.md) and
[reproducible first rollout](docs/en/first-rollout.md).

Release assets contain the wheel, source distribution, source manifest and `SHA256SUMS`.
The manifest hashes the reviewed source export; it is not a model-weight manifest.
Check downloaded assets with `sha256sum --check SHA256SUMS` from the same directory.
