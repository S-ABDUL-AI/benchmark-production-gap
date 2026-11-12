"""
run_analysis.py
Benchmark-Production Gap Analysis Pipeline

Quantifies the gap between AI safety benchmark scores and real-world
harmful output rates using Oaxaca-Blinder decomposition and
Prediction-Powered Inference (PPI).

Usage:
    python run_analysis.py --models vicuna-7b wizardlm-7b koala-7b
                           --sample-size 5000
                           --annotated-sample 200
                           --output-dir ./outputs

Requirements:
    See requirements.txt. Install HarmBench and StrongREJECT separately.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Step 1: Load production data ──────────────────────────────────────────────

def load_lmsys(models: list[str], sample_size: int) -> pd.DataFrame:
    """
    Load and sample LMSYS-Chat-1M from HuggingFace.

    Filters to open-weight models only (required for benchmark re-runs).
    Stratified sample by model and harm category to ensure coverage.

    Args:
        models: List of model names to include (must be open-weight).
        sample_size: Total conversations to sample across all models.

    Returns:
        DataFrame with columns: conversation_id, model, turn, prompt,
        response, harm_category.
    """
    log.info("Loading LMSYS-Chat-1M (requires HF_TOKEN in .env) ...")
    # from datasets import load_dataset
    # ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
    # df = ds.to_pandas()
    raise NotImplementedError("Set HF_TOKEN in .env, then uncomment above.")


def filter_open_weight(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """
    Keep only conversations from open-weight models.

    Closed-weight models (GPT-4, Claude) cannot be re-run on benchmarks,
    so the benchmark-production comparison requires open-weight models only.
    """
    return df[df["model"].isin(models)].copy()


# ── Step 2: Score models on safety benchmarks ─────────────────────────────────

def run_harmbench(models: list[str]) -> pd.DataFrame:
    """
    Run each model through HarmBench and return per-model, per-category scores.

    HarmBench provides ~400 adversarial prompts across harm categories.
    Score = fraction of prompts the model successfully refused.

    Returns:
        DataFrame with columns: model, harm_category, harmbench_score.
    """
    log.info("Running HarmBench evaluations ...")
    # Import and run HarmBench per github.com/centerforaisafety/HarmBench
    raise NotImplementedError("Install HarmBench and implement runner here.")


def run_strongreject(models: list[str]) -> pd.DataFrame:
    """
    Run each model through StrongREJECT and return per-model scores.

    Returns:
        DataFrame with columns: model, harm_category, strongreject_score.
    """
    log.info("Running StrongREJECT evaluations ...")
    # Import and run StrongREJECT per github.com/alexandrasouly/strongreject
    raise NotImplementedError("Install StrongREJECT and implement runner here.")


def merge_benchmark_scores(
    harmbench: pd.DataFrame, strongreject: pd.DataFrame
) -> pd.DataFrame:
    """Combine HarmBench and StrongREJECT into a single benchmark score table."""
    merged = harmbench.merge(strongreject, on=["model", "harm_category"], how="inner")
    merged["benchmark_score"] = merged[["harmbench_score", "strongreject_score"]].mean(axis=1)
    return merged


# ── Step 3: Label production conversations ────────────────────────────────────

def label_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each production conversation as harmful / not harmful using an LLM.

    Uses Claude or GPT-4 as the classifier. Labels are soft predictions
    that will be calibrated via PPI against human annotations.

    Returns:
        df with added column: llm_label (0 = safe, 1 = harmful), llm_confidence.
    """
    log.info("Labeling %d conversations with LLM classifier ...", len(df))
    # import anthropic
    # client = anthropic.Anthropic()
    # for idx, row in df.iterrows():
    #     response = client.messages.create(...)
    #     df.at[idx, "llm_label"] = parse_label(response)
    raise NotImplementedError("Implement LLM labeling loop here.")


def load_human_annotations(path: str) -> pd.DataFrame:
    """
    Load human-annotated ground truth for the PPI calibration subsample.

    Expects a CSV with columns: conversation_id, human_label (0/1).
    Annotate ~200 conversations manually; use inter-rater agreement
    (Cohen's kappa) on a 20% subsample to validate annotation quality.
    """
    log.info("Loading human annotations from %s ...", path)
    return pd.read_csv(path)


def apply_ppi(
    df: pd.DataFrame,
    human_annotations: pd.DataFrame,
) -> dict:
    """
    Apply Prediction-Powered Inference (PPI) to produce a calibrated
    estimate of the production harmful-output rate with valid confidence intervals.

    PPI corrects for systematic LLM labeling bias using the human subsample,
    without requiring full human annotation of the dataset.

    Reference: Angelopoulos et al. (2023), arxiv.org/abs/2301.09633

    Returns:
        dict with keys: ppi_estimate, ci_lower, ci_upper, n_human, n_total.
    """
    log.info("Applying PPI correction ...")
    # from ppi_python import ppi_mean_ci
    # merged = df.merge(human_annotations, on="conversation_id", how="left")
    # labeled = merged[merged["human_label"].notna()]
    # unlabeled = merged[merged["human_label"].isna()]
    # estimate, ci = ppi_mean_ci(
    #     Y=labeled["human_label"].values,
    #     Yhat=labeled["llm_label"].values,
    #     Yhat_unlabeled=unlabeled["llm_label"].values,
    #     alpha=0.05,
    # )
    raise NotImplementedError("Implement PPI calibration here.")


# ── Step 4: Oaxaca-Blinder decomposition ──────────────────────────────────────

def oaxaca_blinder(
    production_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose the benchmark-production gap into composition and structural effects.

    The Oaxaca-Blinder decomposition separates the overall gap (benchmark score
    vs. production harmful-output rate) into:

        Total gap = Composition effect + Within-category (structural) effect

    - Composition effect: gap attributable to differences in what users ask
      vs. what benchmarks test (prompt distribution mismatch)
    - Structural effect: gap that persists even on comparable prompt categories
      (genuine benchmark blind spots)

    Returns:
        DataFrame with columns: model, total_gap, composition_effect,
        structural_effect, composition_pct, structural_pct.
    """
    log.info("Running Oaxaca-Blinder decomposition ...")
    # from statsmodels.formula.api import ols
    # Fit separate regressions for benchmark and production distributions
    # Decompose gap using coefficient and covariate differences
    raise NotImplementedError("Implement Oaxaca-Blinder decomposition here.")


# ── Step 5: Output and visualization ──────────────────────────────────────────

def save_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Save decomposition results and summary statistics to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "gap_decomposition.csv", index=False)
    log.info("Results saved to %s", output_dir / "gap_decomposition.csv")


def plot_gap_heatmap(results: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot a model × harm-category heatmap of benchmark-production gaps.
    Saves as gap_heatmap.png.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = results.pivot(index="model", columns="harm_category", values="total_gap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("Benchmark-Production Gap by Model and Harm Category")
    fig.tight_layout()
    fig.savefig(output_dir / "gap_heatmap.png", dpi=150)
    log.info("Heatmap saved to %s", output_dir / "gap_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark-Production Gap Analysis")
    p.add_argument("--models", nargs="+", default=["vicuna-7b", "wizardlm-7b", "koala-7b"])
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--annotated-sample", type=int, default=200,
                   help="Number of conversations to human-annotate for PPI calibration")
    p.add_argument("--annotations-path", type=str, default="./data/human_annotations.csv")
    p.add_argument("--output-dir", type=str, default="./outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    log.info("=== Benchmark-Production Gap Analysis ===")
    log.info("Models: %s", args.models)
    log.info("Production sample size: %d", args.sample_size)
    log.info("Human annotation sample: %d", args.annotated_sample)

    # 1. Load and filter production data
    production = load_lmsys(args.models, args.sample_size)
    production = filter_open_weight(production, args.models)

    # 2. Score models on safety benchmarks
    harmbench_scores = run_harmbench(args.models)
    strongreject_scores = run_strongreject(args.models)
    benchmark_scores = merge_benchmark_scores(harmbench_scores, strongreject_scores)

    # 3. Label production conversations + PPI calibration
    production = label_with_llm(production)
    human_annotations = load_human_annotations(args.annotations_path)
    ppi_results = apply_ppi(production, human_annotations)
    log.info("PPI estimate: %.3f (95%% CI: %.3f–%.3f)",
             ppi_results["ppi_estimate"],
             ppi_results["ci_lower"],
             ppi_results["ci_upper"])

    # 4. Oaxaca-Blinder decomposition
    gap_results = oaxaca_blinder(production, benchmark_scores)

    # 5. Save and visualize
    save_results(gap_results, output_dir)
    plot_gap_heatmap(gap_results, output_dir)

    log.info("=== Analysis complete. Results in %s ===", output_dir)


if __name__ == "__main__":
    main()
