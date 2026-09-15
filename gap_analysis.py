"""تحلیل شکاف (Gap Analysis): کدام دسته‌های اوتیسم پوشش قوی/ضعیف دارند؟

خروجی:
- توزیع تعداد و درصد هر دسته (top-1).
- میانگین/میانه/انحراف معیار امتیاز top-1 برای هر دسته.
- نسبت نمونه‌های «بااعتماد» (top_score >= threshold).
- رتبه‌بندی قوی/ضعیف و توصیه برای فاز بعدی (داده‌ی مکمل).
- نمودار میله‌ای توزیع + نمودار میانگین امتیاز.
- گزارش Markdown و JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # بدون GUI / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from facts import CATEGORIES, CATEGORY_KEYS


def _category_label(cat: str) -> str:
    return f"{cat}. {CATEGORIES[cat]['title_en']}"


def _label_lists(df: pd.DataFrame) -> list[list[str]]:
    """لیبل‌های multi-label هر نمونه؛ اگر ستون نباشد فقط top-1."""
    tops = df["top_label"].astype(str).tolist()
    if "labels" not in df.columns:
        return [[t] for t in tops]
    out: list[list[str]] = []
    for raw, top in zip(df["labels"], tops):
        if isinstance(raw, (list, tuple)):
            labs = [str(x) for x in raw if x and str(x) != "NONE"]
        elif isinstance(raw, str) and raw.strip():
            labs = [x for x in raw.split("|") if x and x != "NONE"]
        else:
            labs = [top] if top != "NONE" else []
        out.append(labs)
    return out


def build_report(df: pd.DataFrame, scores: np.ndarray,
                 threshold: float = config.LABEL_THRESHOLD) -> dict:
    """ساختن دیکشنری گزارش از DataFrame لیبل‌شده."""
    n = len(df)
    cat_keys = CATEGORY_KEYS

    # امتیاز top-1 هر نمونه و دسته‌ی top
    top_scores = df["top_score"].to_numpy(dtype=float)
    top_cats = df["top_label"].to_numpy()
    multi_lists = _label_lists(df)
    global_means = np.mean(scores, axis=0) if n else np.zeros(len(cat_keys))

    per_cat: dict[str, dict] = {}
    for ci, c in enumerate(cat_keys):
        mask = top_cats == c
        cnt = int(mask.sum())
        sc = top_scores[mask] if cnt else np.array([])
        multi_cnt = sum(1 for labs in multi_lists if c in labs)
        per_cat[c] = {
            "title": CATEGORIES[c]["title_en"],
            "title_fa": CATEGORIES[c]["title_fa"],
            "count": cnt,
            "share": round(cnt / n, 4) if n else 0.0,
            "multi_count": multi_cnt,
            "multi_share": round(multi_cnt / n, 4) if n else 0.0,
            "mean_score": round(float(sc.mean()), 4) if cnt else 0.0,
            "median_score": round(float(np.median(sc)), 4) if cnt else 0.0,
            "std_score": round(float(sc.std()), 4) if cnt else 0.0,
            "confident_share": (
                round(float((sc >= threshold).mean()), 4) if cnt else 0.0
            ),
            "global_mean_score": round(float(global_means[ci]), 4),
        }

    # میانگین امتیاز هر دسته روی کل نمونه‌ها (بدون شرط top-1) — نشان‌دهنده‌ی
    # میزان هم‌خوانی کلی داده‌ها با آن دسته.
    global_cat_score = {
        cat: per_cat[cat]["global_mean_score"] for cat in cat_keys
    }

    # رتبه‌بندی ضعیف→قوی: top-1، سپس پوشش multi-label، سپس شباهت سراسری
    ranking = sorted(
        cat_keys,
        key=lambda c: (
            per_cat[c]["count"],
            per_cat[c]["multi_count"],
            per_cat[c]["global_mean_score"],
        ),
    )

    # با منطق جدید labeler، هر نمونه‌ای که هیچ دسته‌ای threshold را رد نکند
    # top_label=NONE خواهد داشت. n_confident را هم از ستون confident می‌خوانیم
    # (اگر موجود باشد) تا گزارش دقیقاً با خروجی labeler همسو بماند.
    if "confident" in df.columns:
        conf = df["confident"]
        if conf.dtype == bool:
            n_confident = int(conf.sum())
        else:
            normalized = conf.astype(str).str.strip().str.lower()
            n_confident = int(normalized.isin({"true", "1", "yes"}).sum())
    else:
        n_confident = int((top_scores >= threshold).sum())

    n_none = int((df["top_label"].astype(str) == "NONE").sum())
    none_mask = df["top_label"].astype(str).to_numpy() == "NONE"
    none_top_scores = top_scores[none_mask] if n_none else np.array([])

    report = {
        "n_samples": n,
        "threshold": threshold,
        "n_confident": n_confident,
        "confident_rate": round(n_confident / n, 4) if n else 0.0,
        "n_unlabeled_NONE": n_none,
        "none_rate": round(n_none / n, 4) if n else 0.0,
        "none_mean_top_score": (
            round(float(none_top_scores.mean()), 4) if n_none else 0.0
        ),
        "per_category": per_cat,
        "global_mean_score_per_category": global_cat_score,
        "ranking_weak_to_strong": ranking,
        "weakest": ranking[:2],
        "strongest": ranking[-2:],
    }
    return report


def _recommendations(report: dict) -> list[str]:
    recs: list[str] = []
    for c in report["weakest"]:
        pc = report["per_category"][c]
        recs.append(
            f"- **{c}. {pc['title']}**: پوشش ضعیف "
            f"(top-1={pc['count']}, multi={pc.get('multi_count', 0)}, "
            f"share={pc['share']:.1%}, global_mean={pc.get('global_mean_score', 0)}). "
            f"برای فاز after-tuning/RAG داده‌ی مکمل از دیتاست‌های پزشکی/ASD بعدی "
            f"برای این دسته اضافه شود."
        )
    return recs


def render_markdown(report: dict) -> str:
    pc = report["per_category"]
    lines = [
        "# Gap Analysis — MentalChat16K × Autism Facts (A–G)",
        "",
        f"- تعداد نمونه‌ها / samples: **{report['n_samples']}**",
        f"- آستانه‌ی اعتماد / threshold: **{report['threshold']}**",
        f"- نمونه‌های بااعتماد / confident: "
        f"**{report['n_confident']} ({report['confident_rate']:.1%})**",
        f"- بدون لیبل قابل قبول / NONE: "
        f"**{report.get('n_unlabeled_NONE', 0)} "
        f"({report.get('none_rate', 0.0):.1%})**",
        "",
        "## Distribution per category (top-1 and multi-label)",
        "",
        "| Cat | Title (EN) | Top-1 | Share | Multi | Multi% | Mean top-score | Global mean |",
        "|-----|------------|-------|-------|-------|--------|----------------|-------------|",
    ]
    for c in CATEGORY_KEYS:
        d = pc[c]
        mean_top = f"{d['mean_score']}" if d["count"] else "—"
        lines.append(
            f"| {c} | {d['title']} | {d['count']} | {d['share']:.1%} | "
            f"{d.get('multi_count', 0)} | {d.get('multi_share', 0):.1%} | "
            f"{mean_top} | {d.get('global_mean_score', 0)} |"
        )

    # NONE یک دسته‌ی معنایی نیست؛ وضعیت «هیچ دسته‌ای threshold را رد نکرد» است.
    # برای NONE prototype مستقلی نداریم، بنابراین «global mean category similarity»
    # به معنای A–G تعریف نمی‌شود. به‌جای آن، میانگین best/top similarity همان
    # نمونه‌های ردشده را گزارش می‌کنیم تا نمودار و گزارش مقدار معناداری داشته باشند.
    none_count = report.get("n_unlabeled_NONE", 0)
    none_rate = report.get("none_rate", 0.0)
    none_mean = report.get("none_mean_top_score", 0.0)
    lines.append(
        f"| NONE | Below threshold / no accepted category | {none_count} | "
        f"{none_rate:.1%} | {none_count} | {none_rate:.1%} | "
        f"{none_mean if none_count else '—'} | "
        f"{none_mean if none_count else 'N/A'}* |"
    )

    lines += [
        "",
        "## Global mean similarity per category (over all samples)",
        "",
    ]
    for c in CATEGORY_KEYS:
        lines.append(
            f"- {c}. {pc[c]['title']}: {report['global_mean_score_per_category'][c]}"
        )
    if none_count:
        lines.append(
            f"- NONE — mean best similarity among rejected samples: {none_mean}"
        )
    else:
        lines.append(
            "- NONE — N/A (no rejected samples at the current threshold)"
        )
    lines += [
        "",
        "* For NONE, this is not a category-global mean because NONE has no prototype; "
        "it is the mean of each rejected sample's best A–G similarity.",
        "",
        "## Ranking (weak → strong)",
        "",
        " → ".join(report["ranking_weak_to_strong"]),
        "",
        "## Weakest categories & recommendations",
        "",
    ]
    lines += _recommendations(report)
    lines += [
        "",
        "## Strongest categories",
        "",
        f"- **{report['strongest'][-1]}**: {pc[report['strongest'][-1]]['title']}",
        f"- **{report['strongest'][-2]}**: {pc[report['strongest'][-2]]['title']}",
        "",
    ]
    return "\n".join(lines)


def plot_distribution(report: dict, out_path: Path) -> None:
    """رسم توزیع A–G به‌همراه NONE.

    برای A–G خط قرمز «Global mean similarity» رسم می‌شود.
    NONE prototype مستقلی ندارد، پس global mean دسته‌ای برای آن قابل تعریف نیست.
    برای اینکه NONE هم یک شاخص شباهت قابل مشاهده داشته باشد، نقطه‌ی جداگانه‌ای
    با عنوان «NONE mean best similarity» رسم می‌کنیم؛ این مقدار میانگین
    top_score نمونه‌هایی است که هیچ A–G آن‌ها threshold را رد نکرده است.
    """
    cats = CATEGORY_KEYS

    counts = [report["per_category"][c]["count"] for c in cats]
    multi = [report["per_category"][c].get("multi_count", 0) for c in cats]
    means = [
        report["per_category"][c].get("global_mean_score", 0.0)
        for c in cats
    ]
    labels = [_category_label(c) for c in cats]

    none_count = int(report.get("n_unlabeled_NONE", 0))

    # NONE را همیشه نشان می‌دهیم، حتی اگر صفر باشد؛ در نتیجه نمودارهای همه‌ی
    # مدل‌ها ساختار یکسان A–G + NONE خواهند داشت.
    plot_counts = counts + [none_count]
    # وقتی labels=["NONE"] است، NONE همان برچسب پذیرفته‌نشده‌ی نهایی است؛
    # برای نمایش پوشش multi-label نیز همان تعداد را نشان می‌دهیم.
    plot_multi = multi + [none_count]
    none_mean = float(report.get("none_mean_top_score", 0.0)) if none_count else None
    plot_labels = labels + ["NONE — Below threshold"]

    fig, ax1 = plt.subplots(figsize=(12.5, 6.5))
    x = np.arange(len(plot_labels))
    width = 0.38

    bars = ax1.bar(
        x - width / 2,
        plot_counts,
        width,
        color="#4C72B0",
        alpha=0.85,
        label="Top-1 / accepted status",
    )
    bars_m = ax1.bar(
        x + width / 2,
        plot_multi,
        width,
        color="#55A868",
        alpha=0.75,
        label="Multi-label / accepted labels",
    )

    ax1.set_ylabel("Count of samples")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_labels, rotation=20, ha="right", fontsize=9)

    for b, cnt in zip(bars, plot_counts):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for b, cnt in zip(bars_m, plot_multi):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()

    # A–G: تعریف واقعی global mean similarity روی کل نمونه‌ها
    x_cats = x[:len(cats)]
    ax2.plot(
        x_cats,
        means,
        "-o",
        color="#C44E52",
        label="Global mean similarity (A–G)",
    )

    # NONE: چون prototype ندارد، به جای global mean دسته‌ای، میانگین بهترین
    # شباهت A–G برای نمونه‌های ردشده را به صورت یک marker جدا نشان می‌دهیم.
    if none_count and none_mean is not None:
        ax2.scatter(
            [x[-1]],
            [none_mean],
            color="#8172B2",
            marker="D",
            s=52,
            zorder=5,
            label="NONE mean best similarity",
        )
        ax2.annotate(
            f"{none_mean:.3f}",
            (x[-1], none_mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#8172B2",
        )
    else:
        # اگر هیچ NONE نداریم، صریحاً N/A را روی همان ستون نشان بدهیم.
        ax2.annotate(
            "N/A\n(0 NONE)",
            (x[-1], 0.03),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#666666",
        )

    ax2.set_ylabel("Cosine similarity", color="#C44E52")
    similarity_values = list(means)
    if none_count and none_mean is not None:
        similarity_values.append(none_mean)
    ymax = max(similarity_values) + 0.05 if similarity_values else 0.6
    ax2.set_ylim(0, max(0.6, ymax))

    handles, legends = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + h2, legends + l2, loc="upper right", fontsize=8)

    plt.title(
        "MentalChat16K — autism category distribution, NONE rate & similarity"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[gap_analysis] Chart saved -> {out_path}")


def run(
    df: pd.DataFrame,
    scores: np.ndarray,
    output_dir: Path | None = None,
    run_meta: dict | None = None,
) -> dict:
    """اجرای gap analysis و ذخیره‌ی گزارش‌ها.

    Args:
        output_dir: اگر داده شود، خروجی‌ها آنجا نوشته می‌شوند
                    (برای مقایسه‌ی چندمدلی). در غیر این صورت config پیش‌فرض.
        run_meta: متادیتای اختیاری (model, fact_lang, ...) که به JSON اضافه می‌شود.
    """
    report = build_report(df, scores)
    if run_meta:
        report["run"] = run_meta

    if output_dir is None:
        md_path = config.GAP_REPORT_MD
        json_path = config.GAP_REPORT_JSON
        chart_path = config.GAP_CHART_PATH
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "gap_analysis.md"
        json_path = output_dir / "gap_analysis.json"
        chart_path = output_dir / "gap_distribution.png"

    title_extra = ""
    if run_meta:
        title_extra = (
            f"\n\n- run_id: **{run_meta.get('id', '')}**\n"
            f"- model: **{run_meta.get('model_name', run_meta.get('model_key', ''))}**\n"
            f"- fact_lang: **{run_meta.get('fact_lang', '')}**\n"
            f"- dataset: **{run_meta.get('dataset', '')}**\n"
        )
    md = render_markdown(report)
    if title_extra:
        # insert metadata after the first heading block
        parts = md.split("\n", 2)
        if len(parts) >= 3:
            md = parts[0] + "\n" + title_extra + parts[2]
        else:
            md = md + title_extra

    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_distribution(report, chart_path)
    print(f"[gap_analysis] Reports saved -> {md_path}, {json_path}")
    return report
