"""Checkpoint 2 utilities: validate curated multi-hop datasets and produce a
combined success table ready for downstream experiments.

Usage example:
    python checkpoint2_prepare_dataset.py \
        --musique musique_success_20251115_001116.csv \
        --twohop twohopfact_success_20251112_121659.csv \
        --twohop-source path/to/original_twohopfact.csv \
        --output-dir checkpoint2_artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger("checkpoint2")

MUSIQUE_REQUIRED_COLUMNS: Sequence[str] = (
    "row_id",
    "full_question",
    "final_ground_truth",
    "step1_question",
    "step1_ground_truth",
    "step1_response",
    "step2_question",
    "step2_ground_truth",
    "step2_response",
)

TWOHOP_REQUIRED_COLUMNS: Sequence[str] = (
    "uid",
    "r1_prompt",
    "r1_expected_aliases",
    "r1_response",
    "r1_matched_alias",
    "r2_prompt",
    "r2_expected_aliases",
    "r2_response",
    "r2_matched_alias",
    "fact_comp_type",
    "category",
)


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def read_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def _casefold(text: str) -> str:
    return text.casefold() if isinstance(text, str) else ""


def _contains(haystack: str, needle: str) -> bool:
    return _casefold(needle) in _casefold(haystack)


def filter_two_hop_musique(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only two-hop decompositions based on available columns."""
    hop_cols = [col for col in df.columns if col.startswith("step") and col.endswith("_question")]

    # Count sequential step question columns to determine hop count
    def hop_count(row: pd.Series) -> int:
        count = 0
        while f"step{count + 1}_question" in df.columns:
            if pd.notna(row.get(f"step{count + 1}_question")):
                count += 1
            else:
                break
        return count

    mask = df.apply(lambda row: hop_count(row) == 2, axis=1)
    filtered = df.loc[mask].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    if dropped:
        LOGGER.info("Dropped %d MuSiQue rows with hop count != 2", dropped)
    return filtered


def validate_musique(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    for idx, row in df.iterrows():
        row_id = row.get("row_id", f"row_{idx}")
        # Light sanity checks only: presence of questions/answers
        for field in ("step1_question", "step1_ground_truth", "step2_question", "step2_ground_truth"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                issues.append(f"{row_id}: missing or empty {field}")
    return issues


def split_aliases(alias_blob: str) -> List[str]:
    return [alias.strip() for alias in str(alias_blob).split("|") if alias.strip()]


def validate_twohop(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    for idx, row in df.iterrows():
        uid = row.get("uid", f"row_{idx}")
        r1_aliases = split_aliases(row.get("r1_expected_aliases", ""))
        r2_aliases = split_aliases(row.get("r2_expected_aliases", ""))
        r1_match = row.get("r1_matched_alias")
        r2_match = row.get("r2_matched_alias")
        for alias_list_name, aliases in (("r1_expected_aliases", r1_aliases), ("r2_expected_aliases", r2_aliases)):
            if not aliases:
                issues.append(f"{uid}: {alias_list_name} empty")
    return issues


def normalize_musique(df: pd.DataFrame, source: Path) -> List[dict]:
    records: List[dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "dataset": "musique",
                "row_id": row["row_id"],
                "full_question": row["full_question"],
                "final_ground_truth": row["final_ground_truth"],
                "hop_questions": json.dumps(
                    [row["step1_question"], row["step2_question"]], ensure_ascii=False
                ),
                "hop_ground_truths": json.dumps(
                    [row["step1_ground_truth"], row["step2_ground_truth"]],
                    ensure_ascii=False,
                ),
                "hop_responses": json.dumps(
                    [row["step1_response"], row["step2_response"]], ensure_ascii=False
                ),
            }
        )
    return records


def load_twohop_source(fact_csv: Optional[Path]) -> pd.DataFrame:
    """
    Load original TwoHopFact source. If a CSV path is given, load that CSV.
    Otherwise, fall back to loading the Hugging Face 'soheeyang/TwoHopFact' dataset.
    """
    if fact_csv is not None:
        LOGGER.info("Loading TwoHopFact source data from %s", fact_csv)
        if not fact_csv.exists():
            raise FileNotFoundError(f"TwoHopFact source file not found: {fact_csv}")
        return pd.read_csv(fact_csv)

    LOGGER.info("Loading TwoHopFact dataset via Hugging Face API (soheeyang/TwoHopFact)...")
    try:
        from datasets import load_dataset

        ds = load_dataset("soheeyang/TwoHopFact", split="train")
        return ds.to_pandas()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load TwoHopFact source dataset from Hugging Face. "
            "Provide --twohop-source path to a local CSV."
        ) from exc


def merge_twohop_full_question(
    df: pd.DataFrame,
    fact_csv: Optional[Path],
) -> pd.DataFrame:
    """
    Attempt to enrich `df` (the TwoHop success CSV) with 'full_question' and
    'final_ground_truth' from the original TwoHopFact source. If a uid is not
    found in the source, the corresponding fields will be left as None.
    """
    try:
        source_df = load_twohop_source(fact_csv)
    except Exception as exc:
        LOGGER.warning("Could not load TwoHopFact source: %s", exc)
        LOGGER.warning("Skipping full_question / final_ground_truth merge.")
        # Ensure columns exist but return original df unchanged
        df = df.copy()
        if "full_question" not in df.columns:
            df["full_question"] = None
        if "final_ground_truth" not in df.columns:
            df["final_ground_truth"] = None
        return df

    # Columns we might look for in the original dataset
    preferred_cols = [
        "cot.r1(e1).r2(e2).prompt",
        "r2(r1(e1)).prompt",
        "mu.template",
        "e3.value",
        "final_ground_truth",
    ]

    # Make a lookup by uid
    if "uid" not in source_df.columns:
        LOGGER.warning("Source TwoHopFact dataset missing 'uid' column; cannot merge full questions.")
        # still add columns to df to keep downstream stable
        df = df.copy()
        if "full_question" not in df.columns:
            df["full_question"] = None
        if "final_ground_truth" not in df.columns:
            df["final_ground_truth"] = None
        return df

    lookup = source_df.set_index("uid", drop=False)

    missing_uids: List[str] = []
    full_questions: List[Optional[str]] = []
    final_answers: List[Optional[str]] = []

    for uid in df["uid"]:
        if uid in lookup.index:
            src_row = lookup.loc[uid]
            # src_row may be a Series or DataFrame (if duplicate uids) — handle both
            if isinstance(src_row, pd.DataFrame):
                # take the first occurrence
                src_row = src_row.iloc[0]

            # prefer composite prompt fields if present
            full_q = None
            for col in ("r2(r1(e1)).prompt", "cot.r1(e1).r2(e2).prompt", "mu.template"):
                if col in src_row and pd.notna(src_row.get(col)):
                    full_q = src_row.get(col)
                    break

            # fallback to other possible locations
            if not full_q and "full_question" in src_row and pd.notna(src_row.get("full_question")):
                full_q = src_row.get("full_question")

            # final answer ground truth
            answer = None
            for col in ("final_ground_truth", "e3.value"):
                if col in src_row and pd.notna(src_row.get(col)):
                    answer = src_row.get(col)
                    break

            full_questions.append(full_q)
            final_answers.append(answer)
        else:
            missing_uids.append(uid)
            full_questions.append(None)
            final_answers.append(None)

    if missing_uids:
        LOGGER.warning("TwoHopFact full question lookup missing %d uids (examples): %s", len(missing_uids), missing_uids[:5])

    df = df.copy()
    df["full_question"] = full_questions
    df["final_ground_truth"] = final_answers
    return df


def normalize_twohop(df: pd.DataFrame, source: Path) -> List[dict]:
    records: List[dict] = []
    for _, row in df.iterrows():
        hop_ground_truths = [
            json.dumps(split_aliases(row["r1_expected_aliases"]), ensure_ascii=False),
            json.dumps(split_aliases(row["r2_expected_aliases"]), ensure_ascii=False),
        ]
        records.append(
            {
                "dataset": "twohopfact",
                "row_id": row["uid"],
                "full_question": row.get("full_question"),
                "final_ground_truth": row.get("final_ground_truth"),
                "hop_questions": json.dumps(
                    [row["r1_prompt"], row["r2_prompt"]], ensure_ascii=False
                ),
                "hop_ground_truths": json.dumps(
                    [row["r1_expected_aliases"], row["r2_expected_aliases"]],
                    ensure_ascii=False,
                ),
                "hop_responses": json.dumps(
                    [row["r1_response"], row["r2_response"]], ensure_ascii=False
                ),
            }
        )
    return records


def write_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint 2 dataset preparation")
    parser.add_argument("--musique", type=Path, required=True, help="MuSiQue success CSV")
    parser.add_argument("--twohop", type=Path, required=True, help="TwoHopFact success CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoint2_artifacts"),
        help="Directory for combined dataset and validation report",
    )
    parser.add_argument(
        "--twohop-source",
        type=Path,
        default=None,
        help="Optional path to the original TwoHopFact CSV for full questions. "
             "If not provided, the script will try to load the Hugging Face dataset.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    musique_df = read_csv(args.musique)
    twohop_df = read_csv(args.twohop)

    # Filter musique to 2-hop rows
    musique_df = filter_two_hop_musique(musique_df)

    # Attempt to merge full question & final answer into twohop_df (always try)
    twohop_df = merge_twohop_full_question(twohop_df, args.twohop_source)

    # Validate required columns exist BEFORE normalization
    ensure_columns(musique_df, MUSIQUE_REQUIRED_COLUMNS, "MuSiQue")
    ensure_columns(twohop_df, TWOHOP_REQUIRED_COLUMNS, "TwoHopFact")

    musique_issues = validate_musique(musique_df)
    twohop_issues = validate_twohop(twohop_df)

    report_lines = ["CHECKPOINT 2 VALIDATION REPORT", "==============================", ""]
    report_lines.append(f"MuSiQue rows: {len(musique_df)}")
    report_lines.append(f"TwoHopFact rows: {len(twohop_df)}")
    report_lines.append("")
    if musique_issues:
        report_lines.append("MuSiQue issues detected:")
        report_lines.extend(f"  - {issue}" for issue in musique_issues)
    else:
        report_lines.append("MuSiQue: all sanity checks passed")
    report_lines.append("")
    if twohop_issues:
        report_lines.append("TwoHopFact issues detected:")
        report_lines.extend(f"  - {issue}" for issue in twohop_issues)
    else:
        report_lines.append("TwoHopFact: all sanity checks passed")

    combined_records = normalize_musique(musique_df, args.musique)
    combined_records.extend(normalize_twohop(twohop_df, args.twohop))

    combined_df = pd.DataFrame(combined_records)
    combined_path = args.output_dir / "combined_success.csv"
    combined_df.to_csv(combined_path, index=False)
    report_lines.append("")
    report_lines.append(f"Combined success rows: {len(combined_df)}")
    report_lines.append(f"Combined dataset written to {combined_path}")

    report_path = args.output_dir / "validation_report.txt"
    write_report(report_path, report_lines)

    LOGGER.info("Validation report saved to %s", report_path)
    LOGGER.info("Combined dataset saved to %s", combined_path)

    if musique_issues or twohop_issues:
        LOGGER.warning("Issues found during validation; review the report before proceeding.")
    else:
        LOGGER.info("Checkpoint 2 dataset preparation complete with no validation issues.")


if __name__ == "__main__":
    main()
