from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G"]
LABEL_ORDER = CATEGORIES + ["NONE"]

RUNS = {
    "minilm": {
        "fa_en": "outputs/comparison/1_fa_en_minilm",
        "en_en": "outputs/en_en_minilm",
    },
    "bge-m3": {
        "fa_en": "outputs/comparison/2_fa_en_bge-m3",
        "en_en": "outputs/en_en_bge-m3",
    },
    "e5-large": {
        "fa_en": "outputs/comparison/3_fa_en_e5-large",
        "en_en": "outputs/en_en_e5-large",
    },
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/language_comparison")
    return p.parse_args()

def load_run(folder):
    path = Path(folder) / "labeled.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    needed = {"idx","text","top_label","best_label","top_score","labels","confident"}
    needed |= {f"score_{c}" for c in CATEGORIES}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df

def align(fa, en, model):
    fa = fa.copy()
    en = en.copy()
    fa["idx"] = fa["idx"].astype(str)
    en["idx"] = en["idx"].astype(str)
    m = fa.merge(en, on="idx", suffixes=("_fa","_en"), validate="one_to_one")
    if len(m) != len(fa) or len(m) != len(en):
        raise ValueError(f"{model}: sample IDs differ between runs")
    same_text = m["text_fa"].fillna("").astype(str) == m["text_en"].fillna("").astype(str)
    if not same_text.all():
        raise ValueError(f"{model}: input texts differ between runs")
    return m

def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes"})

def corr(a,b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.isclose(a.std(),0) or np.isclose(b.std(),0):
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

def share(s, label):
    return float((s.astype(str) == label).mean())

def summarize_model(model, m):
    fa_scores = m[[f"score_{c}_fa" for c in CATEGORIES]].to_numpy(float)
    en_scores = m[[f"score_{c}_en" for c in CATEGORIES]].to_numpy(float)
    return {
        "model": model,
        "n_samples": len(m),
        "best_label_agreement": float((m["best_label_fa"].astype(str) == m["best_label_en"].astype(str)).mean()),
        "accepted_top_label_agreement": float((m["top_label_fa"].astype(str) == m["top_label_en"].astype(str)).mean()),
        "none_rate_fa_en": share(m["top_label_fa"], "NONE"),
        "none_rate_en_en": share(m["top_label_en"], "NONE"),
        "confident_rate_fa_en": float(as_bool(m["confident_fa"]).mean()),
        "confident_rate_en_en": float(as_bool(m["confident_en"]).mean()),
        "mean_top_score_fa_en": float(m["top_score_fa"].astype(float).mean()),
        "mean_top_score_en_en": float(m["top_score_en"].astype(float).mean()),
        "mean_top_score_delta_en_minus_fa": float((m["top_score_en"].astype(float)-m["top_score_fa"].astype(float)).mean()),
        "mean_abs_category_score_difference": float(np.abs(en_scores-fa_scores).mean()),
        "flattened_score_pearson_r": corr(fa_scores.ravel(), en_scores.ravel()),
    }

def summarize_categories(model, m):
    rows = []
    for c in CATEGORIES:
        sfa = m[f"score_{c}_fa"].to_numpy(float)
        sen = m[f"score_{c}_en"].to_numpy(float)
        d = sen - sfa
        rows.append({
            "model": model,
            "category": c,
            "global_mean_similarity_fa_en": float(sfa.mean()),
            "global_mean_similarity_en_en": float(sen.mean()),
            "delta_global_mean_en_minus_fa": float(d.mean()),
            "mean_abs_score_difference": float(np.abs(d).mean()),
            "score_pearson_r": corr(sfa, sen),
            "best_label_share_fa_en": share(m["best_label_fa"], c),
            "best_label_share_en_en": share(m["best_label_en"], c),
            "accepted_top1_share_fa_en": share(m["top_label_fa"], c),
            "accepted_top1_share_en_en": share(m["top_label_en"], c),
        })
    rows.append({
        "model": model,
        "category": "NONE",
        "global_mean_similarity_fa_en": np.nan,
        "global_mean_similarity_en_en": np.nan,
        "delta_global_mean_en_minus_fa": np.nan,
        "mean_abs_score_difference": np.nan,
        "score_pearson_r": np.nan,
        "best_label_share_fa_en": np.nan,
        "best_label_share_en_en": np.nan,
        "accepted_top1_share_fa_en": share(m["top_label_fa"], "NONE"),
        "accepted_top1_share_en_en": share(m["top_label_en"], "NONE"),
    })
    return rows

def save_transition(model, m, out):
    tab = pd.crosstab(m["top_label_fa"].astype(str), m["top_label_en"].astype(str))
    tab = tab.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    tab.to_csv(out / f"label_transition_{model}.csv", encoding="utf-8-sig")
    row_share = tab.div(tab.sum(axis=1).replace(0,np.nan), axis=0)
    row_share.to_csv(out / f"label_transition_row_share_{model}.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(row_share.fillna(0).to_numpy(float), aspect="auto")
    ax.set_xticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("EN facts × EN dataset")
    ax.set_ylabel("FA facts × EN dataset")
    ax.set_title(f"{model}: accepted-label transitions")
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            v = row_share.iloc[i,j]
            if pd.notna(v) and v > 0:
                ax.text(j,i,f"{v:.0%}",ha="center",va="center",fontsize=8)
    fig.colorbar(im, ax=ax, label="Row share")
    fig.tight_layout()
    fig.savefig(out / f"label_transition_{model}.png", dpi=150)
    plt.close(fig)

def plot_top1(model, cats, out):
    sub = cats[cats["model"] == model].copy()
    x = np.arange(len(sub))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11,6))
    ax.bar(x-w/2, sub["accepted_top1_share_fa_en"]*100, w, label="FA facts × EN dataset")
    ax.bar(x+w/2, sub["accepted_top1_share_en_en"]*100, w, label="EN facts × EN dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["category"])
    ax.set_ylabel("Accepted top-1 share (%)")
    ax.set_title(f"{model}: category distribution by fact language")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"top1_share_{model}.png", dpi=150)
    plt.close(fig)

def plot_global_mean(model, cats, out):
    sub = cats[(cats["model"] == model) & (cats["category"] != "NONE")].copy()
    x = np.arange(len(sub))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11,6))
    ax.bar(x-w/2, sub["global_mean_similarity_fa_en"], w, label="FA facts × EN dataset")
    ax.bar(x+w/2, sub["global_mean_similarity_en_en"], w, label="EN facts × EN dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["category"])
    ax.set_ylabel("Global mean cosine similarity")
    ax.set_title(f"{model}: global mean similarity by fact language")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"global_mean_similarity_{model}.png", dpi=150)
    plt.close(fig)

def plot_summary(summary, out):
    x = np.arange(len(summary))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x-w/2, summary["best_label_agreement"]*100, w, label="Best-label agreement")
    ax.bar(x+w/2, summary["accepted_top_label_agreement"]*100, w, label="Accepted-label agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"])
    ax.set_ylim(0,100)
    ax.set_ylabel("Agreement (%)")
    ax.set_title("Agreement across fact-language conditions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "agreement_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x-w/2, summary["none_rate_fa_en"]*100, w, label="FA facts × EN dataset")
    ax.bar(x+w/2, summary["none_rate_en_en"]*100, w, label="EN facts × EN dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel("NONE rate (%)")
    ax.set_title("NONE rate by fact language")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "none_rate_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(summary["model"], summary["mean_abs_category_score_difference"])
    ax.set_ylabel("Mean absolute A-G score difference")
    ax.set_title("Language sensitivity of semantic scores")
    fig.tight_layout()
    fig.savefig(out / "mean_abs_score_difference_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(summary["model"], summary["flattened_score_pearson_r"])
    ax.set_ylim(0,1)
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation of A-G scores across fact languages")
    fig.tight_layout()
    fig.savefig(out / "score_correlation_by_model.png", dpi=150)
    plt.close(fig)

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"

def fmt4(x):
    return "—" if pd.isna(x) else f"{x:.4f}"

def build_report(summary, cats):
    lines = [
        "# Fact-language comparison",
        "",
        "Comparison: **Persian facts × English dataset (FA_EN)** vs **English facts × English dataset (EN_EN)**.",
        "Within each model, the dataset records are identical and only the fact language changes.",
        "",
        "## Interpretation",
        "",
        "- Raw score / `best_label` metrics are **threshold-independent** and are the main evidence for language sensitivity.",
        "- `NONE`, `confident`, multi-label, and accepted `top_label` metrics are **threshold-dependent**.",
        "- With a shared uncalibrated threshold such as 0.35, threshold-dependent metrics are descriptive, not accuracy measures.",
        "- Without expert gold labels, this analysis can show **stability or change**, but not which language condition is more correct.",
        "",
        "## Overall comparison",
        "",
        "| Model | N | Best-label agreement | Accepted-label agreement | NONE FA_EN | NONE EN_EN | Mean top FA_EN | Mean top EN_EN | Mean abs score diff | Score correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['n_samples'])} | {fmt_pct(r['best_label_agreement'])} | "
            f"{fmt_pct(r['accepted_top_label_agreement'])} | {fmt_pct(r['none_rate_fa_en'])} | "
            f"{fmt_pct(r['none_rate_en_en'])} | {fmt4(r['mean_top_score_fa_en'])} | "
            f"{fmt4(r['mean_top_score_en_en'])} | {fmt4(r['mean_abs_category_score_difference'])} | "
            f"{fmt4(r['flattened_score_pearson_r'])} |"
        )

    lines += [
        "",
        "## Per-category comparison",
        "",
        "| Model | Cat | Global mean FA_EN | Global mean EN_EN | Δ EN−FA | Score r | Best-share FA_EN | Best-share EN_EN | Accepted-share FA_EN | Accepted-share EN_EN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cats[cats["category"] != "NONE"].iterrows():
        lines.append(
            f"| {r['model']} | {r['category']} | {fmt4(r['global_mean_similarity_fa_en'])} | "
            f"{fmt4(r['global_mean_similarity_en_en'])} | {fmt4(r['delta_global_mean_en_minus_fa'])} | "
            f"{fmt4(r['score_pearson_r'])} | {fmt_pct(r['best_label_share_fa_en'])} | "
            f"{fmt_pct(r['best_label_share_en_en'])} | {fmt_pct(r['accepted_top1_share_fa_en'])} | "
            f"{fmt_pct(r['accepted_top1_share_en_en'])} |"
        )

    lines += [
        "",
        "## Recommended reporting language",
        "",
        "At this stage, describe **language sensitivity/stability**, not model accuracy.",
        "After expert-labeled calibration data are available, keep one calibrated threshold per model fixed across FA_EN and EN_EN, and then report Precision, Recall, F1, and Macro-F1 on a held-out test set.",
        "",
    ]
    return "\n".join(lines)

def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    cat_rows = []

    for model, paths in RUNS.items():
        print(f"[compare-language] {model}")
        fa = load_run(paths["fa_en"])
        en = load_run(paths["en_en"])
        m = align(fa, en, model)

        summaries.append(summarize_model(model, m))
        cat_rows.extend(summarize_categories(model, m))

        cols = ["idx","text_fa","top_label_fa","top_label_en","best_label_fa","best_label_en","top_score_fa","top_score_en"]
        cols += [f"score_{c}_fa" for c in CATEGORIES]
        cols += [f"score_{c}_en" for c in CATEGORIES]
        m[cols].to_csv(out / f"paired_samples_{model}.csv", index=False, encoding="utf-8-sig")

        save_transition(model, m, out)

    summary = pd.DataFrame(summaries)
    cats = pd.DataFrame(cat_rows)

    summary.to_csv(out / "language_comparison_summary.csv", index=False, encoding="utf-8-sig")
    cats.to_csv(out / "language_category_comparison.csv", index=False, encoding="utf-8-sig")
    (out / "language_comparison_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    for model in RUNS:
        plot_top1(model, cats, out)
        plot_global_mean(model, cats, out)

    plot_summary(summary, out)
    (out / "language_comparison_report.md").write_text(
        build_report(summary, cats),
        encoding="utf-8"
    )

    print(f"Done -> {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
