# Llama 1B Experimentation Roadmap

## Checkpoint 1 – Environment & Model Access
- **Tasks**
  - Install required Python packages (`torch`, `transformers`, `accelerate`, `pyvene`, `datasets`, etc.).
  - Authenticate with Hugging Face (already completed) and confirm license acceptance.
  - Verify GPU/bfloat16 availability and configure default device settings.
  - Load the base **meta-llama/Llama-3.2-1B** model and tokenizer via Hugging Face.
- **Verification**
  - Run the provided loading script end-to-end without errors.
  - Print diagnostic info: device map, dtype, parameter count summary.
  - Confirm tokenizer pad token handling (falls back to `eos_token`).
  - Script to run: `python checkpoint1_load_model.py --model-id meta-llama/Llama-3.2-1B`
  - Expected console log checklist:
    1. Torch version and detected accelerators (CUDA/MPS) reported.
    2. Preferred dtype announcement (bfloat16/float16/float32) printed.
    3. Tokenizer load message followed by notice about pad token (if set).
    4. Model load completion with total/trainable parameter counts (~1,000M scale).
    5. Device map dump (module → device) or single-device notice.
    6. Forward-pass confirmation line `Forward pass ok: logits shape ...`.
  - Treat any missing item as a failure to investigate before moving to Checkpoint 2.

## Checkpoint 2 – Dataset Assembly & Pre-Test Harness
- **Tasks**
  - Build or reuse the curated two-hop dataset (MuSiQue-style) with columns for Q1, Q2 templates, composite prompt, and gold answers.
  - Integrate placeholder substitution logic to render Q2 with correct context.
  - Implement per-hop evaluation harness (musique pipeline) to measure whether the SLM can answer each sub-question independently.
- **Verification**
  - Spot-check several rows (manual + automated assertions) to ensure placeholders resolve correctly and ground truths match expectations.
  - Execute the harness on a small sample (e.g., 25 rows) and confirm the success/failure CSVs reflect accurate per-hop evaluation.
  - Confirm logging captures per-hop outcomes and refusal detection triggers.
  - Script to run: `python checkpoint2_prepare_dataset.py --musique <musique_success.csv> --twohop <twohop_success.csv>`
  - Expected outputs under `checkpoint2_artifacts/`:
    1. `validation_report.txt` summarising row counts and any detected issues.
    2. `combined_success.csv` with normalized schema (dataset tag, hop questions/answers/responses).
  - Treat any reported issue as blocking; resolve before proceeding to Checkpoint 3.

## Checkpoint 3 – Baseline Reproduction (No Interventions)
- **Tasks**
  - Using the HF-loaded model (without pyvene), reproduce the known failure pattern: Q1 pass, Q2 pass, composite fail.
  - Implement failure taxonomy labeling (F1/F2/F3) for each row meeting the failure condition.
- **Verification**
  - Run on an initial subset (e.g., 100 rows) and ensure failure distribution statistics match expectations (record counts for each failure class).
  - Persist baseline results to a structured table (CSV/Parquet) for later comparison.
  - Script to run: `python checkpoint3_baseline.py --dataset checkpoint2_artifacts/combined_success.csv --sample-size 100`
  - Expected outputs under `checkpoint3_artifacts/`:
    1. `baseline_results.csv` capturing per-row prompt, expected answer, HF answer/label, Groq judge verdict/reason, and final label.
    2. `baseline_summary.json` logging raw HF counts, final labels after judge overrides, and judge statistics (checked/confirmed/overruled/errors).
  - Ensure `GROQ_API_KEY` is set in `.env`; the script calls Groq Judge (`openai/gpt-oss-120b`) on each HF failure to confirm correctness.
  - Review any rows where the judge overruled the HF label to verify that “CORRECT” verdicts reflect true equivalence (e.g., synonyms, duplicated phrasing).

## Checkpoint 4 – pyvene Integration
- **Tasks**
  - Install and import `pyvene`; construct an `IntervenableConfig` targeting key Llama decoder components.
  - Wrap the HF model with `pv.IntervenableModel`, preserving generation parity with the base model.
- **Verification**
  - Run identical prompts through both the raw model and the pyvene-wrapped model; assert identical token outputs.
  - Confirm no runtime errors when calling `pv_model.generate` without interventions.

## Checkpoint 5 – Diagnostic Module A (Logit Lens)
- **Tasks**
  - Configure `CollectIntervention` capturing per-layer hidden states at the final token.
  - Project each collected state through the unembedding matrix to compute the rank of the correct intermediate answer token.
  - Generate plots of rank vs. layer for representative failure cases.
- **Verification**
  - Produce and archive a plot showing the predicted “fade-out” rank curve.
  - Validate that extracted ranks correspond to the expected token IDs (sanity check on tokenization).

## Checkpoint 6 – Diagnostic Module B (Back-Patching / Hopping Too Late)
- **Tasks**
  - Implement VanillaIntervention swaps from late to early layers for composite prompts.
  - Sweep a matrix of source/target layers and record correction rates for failed cases.
- **Verification**
  - Generate a table or heatmap summarizing correction percentages.
  - Log examples where interventions flip failures to successes (and any regressions).

## Checkpoint 7 – Mitigation Module (Memory Injection)
- **Tasks**
  - Design intervention schemes that “inject” intermediate answer representations during composite generation.
  - Evaluate impact on composite accuracy and track side effects on other rows.
- **Verification**
  - Compare pre/post success metrics; ensure improvements are statistically meaningful.
  - Archive qualitative transcripts demonstrating corrected reasoning chains.

