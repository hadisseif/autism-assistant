"""اجرای ماتریس ۵ پیکربندی و ساخت گزارش مقایسه‌ای.

پیکربندی‌ها (config.COMPARISON_RUNS) — همه از فکت فارسی استفاده می‌کنند:
1. FA facts × EN dataset × MiniLM   (rerun of task 1; facts may have changed)
2. FA facts × EN dataset × bge-m3
3. FA facts × EN dataset × multilingual-e5-large
4. FA facts × FA dataset × bge-m3   (after GPT-4+ translation)
5. FA facts × FA dataset × multilingual-e5-large

اجرا:
    python compare_runs.py --limit 100
    python compare_runs.py --limit 0          # کل دیتاست
    python compare_runs.py --only 1,2         # فقط بعضی runها
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from facts import CATEGORY_KEYS
from main import run_pipeline


def _reconfigure_stdio_utf8() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_reconfigure_stdio_utf8()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-way embedding comparison.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample limit per run (0 = all). Default: config.SAMPLE_LIMIT",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override Persian dataset path",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated run numbers 1-5, e.g. 1,2,3",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its gap_analysis.json already exists",
    )
    return p.parse_args()


def selected_runs(only: str | None) -> list[dict]:
    """Parse `--only 1,2,3` into comparison run configs."""
    runs = list(config.COMPARISON_RUNS)
    if not only:
        return runs
    wanted = {int(x.strip()) for x in only.split(",") if x.strip()}
    selected = []
    for i, run in enumerate(runs, start=1):
        if i in wanted:
            selected.append(run)
    if not selected:
        raise ValueError(f"No runs matched --only={only!r}")
    return selected


# Back-compat alias
_selected_runs = selected_runs


def summarize_run(report: dict, run_cfg: dict) -> dict:
    pc = report["per_category"]
    return {
        "id": run_cfg["id"],
        "title": run_cfg["title"],
        "model_key": run_cfg["model_key"],
        "fact_lang": run_cfg["fact_lang"],
        "dataset": run_cfg["dataset"],
        "n_samples": report["n_samples"],
        "confident_rate": report["confident_rate"],
        "none_count": report.get("n_unlabeled_NONE", 0),
        "none_rate": report.get("none_rate", 0.0),
        "mean_top_score": round(
            float(np.mean([pc[c]["mean_score"] for c in CATEGORY_KEYS
                           if pc[c]["count"] > 0] or [0.0])),
            4,
        ),
        "ranking_weak_to_strong": report["ranking_weak_to_strong"],
        "counts": {c: pc[c]["count"] for c in CATEGORY_KEYS},
        "shares": {c: pc[c]["share"] for c in CATEGORY_KEYS},
        "multi_counts": {c: pc[c].get("multi_count", 0) for c in CATEGORY_KEYS},
        "multi_shares": {c: pc[c].get("multi_share", 0.0) for c in CATEGORY_KEYS},
        "mean_scores": {c: pc[c]["mean_score"] for c in CATEGORY_KEYS},
        "global_mean_score_per_category": report["global_mean_score_per_category"],
    }


# Back-compat alias
_summarize_run = summarize_run


def build_comparison_table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        row = {
            "run": s["id"],
            "title": s["title"],
            "model": s["model_key"],
            "fact_lang": s["fact_lang"],
            "n_samples": s["n_samples"],
            "confident_rate": s["confident_rate"],
            "none_count": s.get("none_count", 0),
            "none_rate": s.get("none_rate", 0.0),
            "mean_top_score": s["mean_top_score"],
            "weakest": " > ".join(s["ranking_weak_to_strong"][:2]),
            "strongest": " > ".join(s["ranking_weak_to_strong"][-2:]),
        }
        for c in CATEGORY_KEYS:
            row[f"count_{c}"] = s["counts"][c]
            row[f"share_{c}"] = s["shares"][c]
            row[f"multi_{c}"] = s.get("multi_counts", {}).get(c, 0)
            row[f"mean_{c}"] = s["mean_scores"][c]

        # NONE دسته‌ی معنایی نیست، اما برای مقایسه‌ی thresholdها باید
        # count/share آن در جدول نهایی قابل مشاهده باشد.
        row["count_NONE"] = s.get("none_count", 0)
        row["share_NONE"] = s.get("none_rate", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison_markdown(summaries: list[dict], table: pd.DataFrame) -> str:
    top1_keys = list(CATEGORY_KEYS) + ["NONE"]

    lines = [
        "# Embedding comparison",
        "",
        "Comparison of semantic labeling runs against autism facts (A–G).",
        "`NONE` means that no A–G category passed the configured threshold.",
        "",
        "## Runs",
        "",
        "| # | ID | Facts | Dataset | Model | Confident% | NONE% | Mean top-score | Weak→Strong |",
        "|---|----|-------|---------|-------|------------|-------|----------------|-------------|",
    ]
    for i, s in enumerate(summaries, start=1):
        lines.append(
            f"| {i} | `{s['id']}` | {s['fact_lang']} | {s['dataset']} | "
            f"{s['model_key']} | {s['confident_rate']:.1%} | "
            f"{s.get('none_rate', 0.0):.1%} | {s['mean_top_score']} | "
            f"{' → '.join(s['ranking_weak_to_strong'])} |"
        )

    lines += [
        "",
        "## Category share (top-1 %) — including NONE",
        "",
        "| Run | " + " | ".join(top1_keys) + " |",
        "|-----|" + "|".join(["------"] * len(top1_keys)) + "|",
    ]
    for s in summaries:
        values = [s["shares"][c] for c in CATEGORY_KEYS]
        values.append(s.get("none_rate", 0.0))
        shares = " | ".join(f"{v:.1%}" for v in values)
        lines.append(f"| `{s['id']}` | {shares} |")

    lines += [
        "",
        "## Category multi-label share (%) — including NONE status",
        "",
        "| Run | " + " | ".join(top1_keys) + " |",
        "|-----|" + "|".join(["------"] * len(top1_keys)) + "|",
    ]
    for s in summaries:
        multi = s.get("multi_shares") or {c: 0.0 for c in CATEGORY_KEYS}
        values = [multi[c] for c in CATEGORY_KEYS]
        # NONE تنها زمانی رخ می‌دهد که هیچ A–G پذیرفته نشده باشد؛ در labels نیز
        # به صورت [NONE] ثبت می‌شود، پس share آن همان none_rate است.
        values.append(s.get("none_rate", 0.0))
        shares = " | ".join(f"{v:.1%}" for v in values)
        lines.append(f"| `{s['id']}` | {shares} |")

    lines += [
        "",
        "## Category global mean similarity",
        "",
        "`NONE` در این جدول وجود ندارد، چون prototype/embedding مستقلی برای NONE تعریف نشده است.",
        "",
        "| Run | " + " | ".join(CATEGORY_KEYS) + " |",
        "|-----|" + "|".join(["------"] * len(CATEGORY_KEYS)) + "|",
    ]
    for s in summaries:
        gmeans = s.get("global_mean_score_per_category") or s["mean_scores"]
        means = " | ".join(str(gmeans[c]) for c in CATEGORY_KEYS)
        lines.append(f"| `{s['id']}` | {means} |")

    lines += [
        "",
        "## Notes for interpretation",
        "",
        "- Runs 1–3 use **Persian facts × English dataset** to compare three encoders "
        "on the original MentalChat16K.",
        "- Runs 4–5 use **Persian facts × Persian dataset** after translation "
        "(matched-language condition).",
        "- Labeling uses **patient-only** text by default.",
        "- `NONE` is not an eighth autism category. It is a rejection status: no A–G "
        "score reached the configured threshold.",
        "- Because cosine-score scales differ across embedding models, `NONE%` and "
        "confident_rate should be interpreted together with model-specific threshold calibration.",
        "",
        f"Full numeric table also saved as `comparison_summary.csv` "
        f"({len(table)} rows).",
        "",
    ]
    return "\n".join(lines)


def plot_comparison(summaries: list[dict], out_path: Path) -> None:
    """Top-1 share را برای A–G به‌همراه NONE بین runها مقایسه می‌کند."""
    if not summaries:
        return

    plot_keys = list(CATEGORY_KEYS) + ["NONE"]
    n_runs = len(summaries)
    x = np.arange(len(plot_keys))
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for i, s in enumerate(summaries):
        shares = [s["shares"][c] * 100 for c in CATEGORY_KEYS]
        shares.append(s.get("none_rate", 0.0) * 100)

        ax.bar(
            x + i * width,
            shares,
            width=width,
            label=s["id"],
        )

    ax.set_xticks(x + width * (n_runs - 1) / 2)
    ax.set_xticklabels(plot_keys)
    ax.set_ylabel("Top-1 / NONE share (%)")
    ax.set_title("Accepted category distribution across comparison runs (A–G + NONE)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[compare] Chart saved -> {out_path}")


def main() -> int:
    args = parse_args()
    limit = args.limit if args.limit is not None else config.SAMPLE_LIMIT
    runs = selected_runs(args.only)

    fa_path = Path(args.dataset_path) if args.dataset_path else config.PERSIAN_DATASET_PATH
    needs_fa = any(r["dataset"] == "fa" for r in runs)
    if needs_fa and not fa_path.exists():
        print(
            f"[compare] ERROR: Persian dataset missing: {fa_path}\n"
            "1) python export_for_translation.py --limit 500\n"
            "2) Translate with prompts/semantic_persian_translation.txt (GPT-4+)\n"
            "3) Save as data/mentalchat16k_fa.jsonl\n"
            "Note: runs 1-3 use the English dataset and do not need this file.",
            file=sys.stderr,
        )
        return 2

    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for run_cfg in runs:
        run_dir = config.COMPARISON_DIR / run_cfg["id"]
        report_path = run_dir / "gap_analysis.json"
        print("\n" + "#" * 72)
        print(f"# {run_cfg['title']}")
        print("#" * 72)

        if args.skip_existing and report_path.exists():
            print(f"[compare] Skipping existing run -> {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            ds = run_cfg["dataset"]
            ds_path = fa_path if ds == "fa" else None
            report = run_pipeline(
                limit=limit,
                model_key=run_cfg["model_key"],
                fact_lang=run_cfg["fact_lang"],
                dataset=ds,
                dataset_path=ds_path,
                output_dir=run_dir,
                run_meta={
                    "id": run_cfg["id"],
                    "title": run_cfg["title"],
                },
            )
        summaries.append(summarize_run(report, run_cfg))

    table = build_comparison_table(summaries)
    csv_path = config.COMPARISON_DIR / "comparison_summary.csv"
    json_path = config.COMPARISON_DIR / "comparison_summary.json"
    md_path = config.COMPARISON_DIR / "comparison_report.md"
    chart_path = config.COMPARISON_DIR / "comparison_shares.png"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(
        render_comparison_markdown(summaries, table), encoding="utf-8"
    )
    plot_comparison(summaries, chart_path)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Report: {md_path}")
    print(f"Table:  {csv_path}")
    print(f"Chart:  {chart_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
