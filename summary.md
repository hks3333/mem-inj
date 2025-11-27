# Project Progress Summary

## Dataset Curation & Preprocessing

### `musique/musique.py`
- Downloads MuSiQue splits via `datasets` and filters to two-hop questions.
- Uses an Ollama-served Llama 3.2 1B model to answer each hop independently.
- Applies refusal and generic-text heuristics, normalizes answers, and checks alias matches.
- Keeps only rows where both hop answers are correct; writes success/failure CSVs plus logs and timing metadata.

### `twohopfact/twohopfact.py`
- Loads TwoHopFact data and parses nested alias structures robustly.
- Reuses an Ollama client to query each hop with deterministic prompts.
- Validates hop responses via normalized alias matching; records reasons for any failure (missing prompts, timeouts, wrong alias, etc.).
- Produces timestamped success CSV used in Checkpoint 2 alongside audit logs.

### `checkpoint2_prepare_dataset.py`
- Ingests the MuSiQue and TwoHopFact “success” CSVs and harmonizes schema (dataset tag, hop prompts/answers, composite question).
- Performs sanity checks: required columns, non-empty strings, hop alignment.
- Emits `combined_success.csv` plus `validation_report.txt` summarising counts and warnings.

## Checkpoint Progress

### Checkpoint 1 – Environment & Model Access (`checkpoint1_load_model.py`)
- Detects available accelerators and picks optimal dtype (bf16/float16/float32).
- Loads `meta-llama/Llama-3.2-1B`, configures tokenizer padding, prints parameter counts/device map.
- Runs a forward pass to confirm the model is ready for downstream checkpoints.

### Checkpoint 2 – Dataset Assembly (`checkpoint2_prepare_dataset.py`)
- Merges the curated success rows from MuSiQue and TwoHopFact into a unified evaluation table.
- Normalizes columns, resolves placeholders, and saves artifacts for later checkpoints.

### Checkpoint 3 – Baseline Evaluation (`checkpoint3_baseline.py`)
- Runs the base Llama 1B model over composite questions (batch generation with deterministic decoding).
- Labels each answer with a custom taxonomy: success, F1_refusal, F2_partial, F3_other.
- Optionally calls Groq Judge to adjudicate failures and writes detailed CSV plus summary JSON.

### Checkpoint 4 – Pyvene Integration (`checkpoint4_pyvene_integration.py`)
- Builds a `pv.IntervenableModel` targeting each layer’s MLP output.
- Verifies generation parity between raw HF model and pyvene-wrapped model across canned prompts.
- Saves parity metrics to `checkpoint4_artifacts`.

### Checkpoint 5 – Logit Lens Diagnostic (`checkpoint5_logit_lens.py`)
- Reconfigures pyvene with `CollectIntervention` on each block output.
- For failure cases (default F2/F3), captures last-token residuals, projects through the unembedding matrix, and logs per-layer rank/logit of the intermediate-hop token.
- Outputs `logit_lens_summary.csv`, `logit_lens_metadata.json`, and optional fade-out plots.

## How to Run Each Script

### MuSiQue Hop Evaluation
```bash
python musique/musique.py \
  --output-dir musique_results \
  --model llama3.2:1b \
  --base-url http://localhost:11434
```
Mandatory args: `--output-dir`, `--model`, `--base-url` (others default: `--sample-size`, `--max-workers`, `--shuffle`, etc.).

### TwoHopFact Hop Evaluation
```bash
python twohopfact/twohopfact.py \
  --output-dir twohop_results \
  --model llama3.2:1b \
  --base-url http://localhost:11434
```
Mandatory args: `--output-dir`, `--model`, `--base-url` (additional flags: `--sample-size`, `--max-workers`).

### Checkpoint 1 – Base Model Load
```bash
python checkpoint1_load_model.py \
  --model-id meta-llama/Llama-3.2-1B \
  --revision <optional_commit>
```
Mandatory args: none; `--model-id` defaults to the project model.

### Checkpoint 2 – Dataset Assembly
```bash
python checkpoint2_prepare_dataset.py \
  --musique musique_success_YYYYMMDD_HHMMSS.csv \
  --twohop twohopfact_success_YYYYMMDD_HHMMSS.csv \
  --output-dir checkpoint2_artifacts
```
Mandatory args: `--musique`, `--twohop`; optional `--output-dir`, `--sample-size`.

### Checkpoint 3 – Baseline Evaluation
```bash
python checkpoint3_baseline.py \
  --dataset checkpoint2_artifacts/combined_success.csv \
  --output-dir checkpoint3_artifacts \
  --sample-size 100
```
Mandatory args: `--dataset`; optional `--model-id`, `--output-dir`, `--sample-size`, `--batch-size`, `--max-tokens`.

### Checkpoint 4 – Pyvene Parity
```bash
python checkpoint4_pyvene_integration.py \
  --model-id meta-llama/Llama-3.2-1B \
  --num-test-prompts 10 \
  --output-dir checkpoint4_artifacts
```
Mandatory args: none (defaults cover model id, prompt count, output dir).

### Checkpoint 5 – Logit Lens Diagnostic
```bash
python checkpoint5_logit_lens.py \
  --dataset checkpoint2_artifacts/combined_success.csv \
  --baseline-results checkpoint3_artifacts/baseline_results.csv \
  --output-dir checkpoint5_artifacts \
  --sample-size 10 \
  --generate-plots
```
Mandatory args: none; key optional flags include `--dataset`, `--baseline-results`, `--target-labels`, `--sample-size`, `--generate-plots`.

### Upcoming Checkpoints
- **Checkpoint 6 (planned command sketch):**
  ```bash
  python checkpoint6_back_patching.py \
    --dataset checkpoint3_artifacts/baseline_results.csv \
    --output-dir checkpoint6_artifacts
  ```
  (Will require `--dataset` and `--output-dir` once implemented.)

- **Checkpoint 7 (planned command sketch):**
  ```bash
  python checkpoint7_memory_injection.py \
    --dataset checkpoint3_artifacts/baseline_results.csv \
    --output-dir checkpoint7_artifacts
  ```
  (Anticipated mandatory args: `--dataset`, `--output-dir`, plus intervention configuration flags.)

### Upcoming Work
- **Checkpoint 6 – Back-Patching:** Swap late-layer activations into earlier layers to test whether overwriting drives failures; summarize correction heatmaps.
- **Checkpoint 7 – Memory Injection:** Inject intermediate answers during generation to mitigate failures; evaluate accuracy lift and side effects on held-out cases.
