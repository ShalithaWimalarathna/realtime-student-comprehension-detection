"""
KIU Comprehension System — Statistical Analysis & Chart Generator
=================================================================
Generates:
    Figure 5.2  → sus_score_distribution.png   (SUS score bar chart)
    Figure 5.3  → correlation_chart.png         (Engaged % vs Comprehension scatter)
    Table 5.3   → correlation_results.txt        (all Pearson/Spearman values)
    Full report → analysis_report.txt

HOW TO RUN:
    1. Make sure you have all participant SUS CSV files in sus_responses/
    2. Make sure you have all session log CSVs in logs/
    3. Run:  python analyze_results.py
    4. All output files saved to results/ folder

INPUTS NEEDED:
    sus_responses/MASTER_responses.csv  (run merge_sus_responses.py first)
    logs/ALL_SESSIONS.csv               (copy/merge your session logs here)

    If you only have one session log (single participant test):
    just copy logs/session_log.csv to logs/ALL_SESSIONS.csv
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Try importing scipy for statistics ───────────────────────────────────────
try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not installed. Run: pip install scipy")
    print("       Correlation analysis will be skipped.")

OUTPUT_DIR  = "results"
SUS_FILE    = "sus_responses/MASTER_responses.csv"
LOG_FILE    = "logs/ALL_SESSIONS.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows, None


def safe_float(val, default=None):
    try:
        return float(str(val).strip())
    except Exception:
        return default


def sus_grade(score):
    if score >= 85:  return "A — Excellent"
    if score >= 72:  return "B — Good"
    if score >= 52:  return "C — Acceptable"
    if score >= 38:  return "D — Poor"
    return "F — Unacceptable"


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  KIU Research — Statistical Analysis & Chart Generator")
print("="*60)

sus_rows, sus_err = load_csv(SUS_FILE)
log_rows, log_err = load_csv(LOG_FILE)

# ── Fallback: try single session log if ALL_SESSIONS not found ───────────────
if log_rows is None:
    log_rows, log_err2 = load_csv("logs/session_log.csv")
    if log_rows:
        print("[INFO] Using logs/session_log.csv (single session)")
    else:
        print(f"[WARN] {log_err}")

if sus_rows is None:
    print(f"[WARN] {sus_err}")
    print("       Run merge_sus_responses.py first to create MASTER_responses.csv")

print(f"\n  SUS responses loaded  : {len(sus_rows) if sus_rows else 0} participants")
print(f"  Session logs loaded   : {len(log_rows)  if log_rows  else 0} readings")


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACT SUS SCORES
# ─────────────────────────────────────────────────────────────────────────────
sus_scores         = []
pre_stress_vals    = []
post_stress_vals   = []
comprehension_vals = []
engagement_vals    = []
participant_ids    = []

if sus_rows:
    for row in sus_rows:
        pid   = row.get("participant_id", "").strip()
        score = safe_float(row.get("sus_score",""))
        pre   = safe_float(row.get("pre_stress",""))
        post  = safe_float(row.get("post_stress",""))
        comp  = safe_float(row.get("comprehension_self",""))
        eng   = safe_float(row.get("engagement_self",""))

        if score is not None:
            sus_scores.append(score)
            participant_ids.append(pid if pid else f"P{len(sus_scores)}")
        if pre  is not None: pre_stress_vals.append(pre)
        if post is not None: post_stress_vals.append(post)
        if comp is not None: comprehension_vals.append(comp)
        if eng  is not None: engagement_vals.append(eng)


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACT SESSION DETECTION PERCENTAGES PER PARTICIPANT
# ─────────────────────────────────────────────────────────────────────────────
# Build per-participant summary: engaged%, confused%, bored%
session_summary = {}   # pid → {engaged_pct, confused_pct, bored_pct}

if log_rows:
    from collections import defaultdict
    pid_readings = defaultdict(lambda: {"Engaged":0,"Confused":0,"Bored":0,"total":0})

    for row in log_rows:
        # Try to get participant ID from log
        # If your log has student_id, we use it; otherwise group all as one
        pid  = row.get("participant_id", row.get("student_id","P1"))
        state = row.get("comprehension_state", row.get("state","Unknown"))
        if state in ["Engaged","Confused","Bored"]:
            pid_readings[pid][state] += 1
            pid_readings[pid]["total"] += 1

    for pid, counts in pid_readings.items():
        total = max(counts["total"], 1)
        session_summary[pid] = {
            "engaged_pct":  counts["Engaged"]  / total * 100,
            "confused_pct": counts["Confused"] / total * 100,
            "bored_pct":    counts["Bored"]    / total * 100,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5.2 — SUS SCORE DISTRIBUTION BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_52():
    if not sus_scores:
        print("\n[SKIP] Figure 5.2 — no SUS scores found")
        return

    print("\n[Generating] Figure 5.2 — SUS Score Distribution...")

    n     = len(sus_scores)
    mean  = np.mean(sus_scores)
    std   = np.std(sus_scores)
    grade = sus_grade(mean)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Colour bars by grade
    colors = []
    for s in sus_scores:
        if s >= 85:   colors.append("#22c55e")  # excellent — green
        elif s >= 72: colors.append("#6366f1")  # good — blue
        elif s >= 52: colors.append("#f59e0b")  # acceptable — amber
        else:         colors.append("#ef4444")  # poor — red

    labels = participant_ids[:n] if participant_ids else [f"P{i+1}" for i in range(n)]
    bars = ax.bar(labels, sus_scores, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.6, zorder=3)

    # Mean line
    ax.axhline(y=mean, color="#1e293b", linewidth=1.8, linestyle="--",
               label=f"Mean = {mean:.1f} ({grade})", zorder=4)

    # Grade zones
    zone_data = [
        (85, 100, "#22c55e", "Excellent (≥85)"),
        (72, 85,  "#6366f1", "Good (72–84)"),
        (52, 72,  "#f59e0b", "Acceptable (52–71)"),
        (0,  52,  "#ef4444", "Poor (<52)"),
    ]
    for lo, hi, col, lbl in zone_data:
        ax.axhspan(lo, hi, alpha=0.06, color=col, zorder=1)

    # Value labels on bars
    for bar, score in zip(bars, sus_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{score:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1e293b")

    ax.set_xlabel("Participant ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("SUS Score (0–100)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"System Usability Scale (SUS) Scores — All Participants\n"
        f"n={n}  |  Mean={mean:.1f}  |  SD={std:.1f}  |  Grade: {grade}",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 38, 52, 72, 85, 100])
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right")

    # Legend for grade colours
    patches = [mpatches.Patch(color=c, label=l, alpha=0.8) for _,_,c,l in zone_data]
    ax.legend(handles=[ax.get_legend_handles_labels()[0][0]] + patches,
              labels=[f"Mean = {mean:.1f} ({grade})"] + [l for _,_,_,l in zone_data],
              fontsize=9, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sus_score_distribution.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    print(f"  Mean SUS: {mean:.1f}  |  SD: {std:.1f}  |  Grade: {grade}")
    return mean, std, grade, n


# ─────────────────────────────────────────────────────────────────────────────
#  TABLE 5.3 + FIGURE 5.3 — CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def generate_correlations_and_figure53():
    if not SCIPY_OK:
        print("\n[SKIP] Correlation analysis — scipy not installed")
        return []

    if not session_summary:
        print("\n[SKIP] Table 5.3 / Figure 5.3 — no session detection data found")
        return []

    if not sus_rows:
        print("\n[SKIP] Table 5.3 / Figure 5.3 — no SUS response data found")
        return []

    print("\n[Generating] Table 5.3 — Correlation Analysis...")

    # ── Match session detection data with SUS questionnaire data ─────────────
    matched = []
    for row in sus_rows:
        pid  = row.get("participant_id","").strip()
        comp = safe_float(row.get("comprehension_self",""))
        eng  = safe_float(row.get("engagement_self",""))
        post = safe_float(row.get("post_stress",""))
        sus  = safe_float(row.get("sus_score",""))

        # Look for matching detection session
        det = session_summary.get(pid) or session_summary.get(
            pid.replace("KIU-","").lstrip("0") or pid
        )
        # Fallback: if only one participant in session summary, use that
        if det is None and len(session_summary) == 1:
            det = list(session_summary.values())[0]

        if det and comp is not None:
            matched.append({
                "pid":          pid,
                "engaged_pct":  det["engaged_pct"],
                "confused_pct": det["confused_pct"],
                "bored_pct":    det["bored_pct"],
                "comprehension":comp,
                "engagement":   eng,
                "post_stress":  post,
                "sus_score":    sus,
            })

    if len(matched) < 3:
        print(f"  [WARN] Only {len(matched)} matched participants found.")
        print("  Need at least 3 for meaningful correlation.")
        print("  Generating chart with available data anyway...")
        if len(matched) == 0:
            print("  No matches — check that participant_id in SUS CSV")
            print("  matches student_id / participant_id in session log CSV.")
            return []

    n = len(matched)
    print(f"  Matched {n} participant(s) for correlation")

    # ── Correlation pairs ─────────────────────────────────────────────────────
    pairs = [
        ("engaged_pct",  "comprehension", "Detected Engaged % vs Self-reported Comprehension"),
        ("confused_pct", "comprehension", "Detected Confused % vs Self-reported Comprehension"),
        ("engaged_pct",  "engagement",    "Detected Engaged % vs Self-reported Engagement"),
        ("confused_pct", "post_stress",   "Detected Confused % vs Post-session Stress"),
        ("sus_score",    "comprehension", "SUS Score vs Self-reported Comprehension"),
    ]

    results = []
    for x_key, y_key, label in pairs:
        xs = [m[x_key] for m in matched if m.get(x_key) is not None and m.get(y_key) is not None]
        ys = [m[y_key] for m in matched if m.get(x_key) is not None and m.get(y_key) is not None]
        if len(xs) < 3:
            results.append({"label":label,"pr":"N/A","pp":"N/A","sr":"N/A","sp":"N/A","sig":"n/a","n":len(xs),"xs":xs,"ys":ys})
            continue
        pr, pp = stats.pearsonr(xs, ys)
        sr, sp = stats.spearmanr(xs, ys)
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "ns"
        results.append({
            "label":label,
            "pr":round(pr,4),"pp":round(pp,4),
            "sr":round(sr,4),"sp":round(sp,4),
            "sig":sig,"n":len(xs),"xs":xs,"ys":ys
        })

    # ── Pre vs post stress t-test ─────────────────────────────────────────────
    ttest_result = None
    pre_vals  = [m["engaged_pct"] for m in matched if "post_stress" in m]  # placeholder
    pre_s  = [safe_float(r.get("pre_stress",""))  for r in sus_rows if safe_float(r.get("pre_stress",""))  is not None]
    post_s = [safe_float(r.get("post_stress","")) for r in sus_rows if safe_float(r.get("post_stress","")) is not None]
    if len(pre_s) >= 3 and len(post_s) >= 3:
        min_len = min(len(pre_s), len(post_s))
        t_stat, t_p = stats.ttest_rel(pre_s[:min_len], post_s[:min_len])
        ttest_result = {
            "pre_mean":  round(np.mean(pre_s),2),
            "pre_std":   round(np.std(pre_s),2),
            "post_mean": round(np.mean(post_s),2),
            "post_std":  round(np.std(post_s),2),
            "t":         round(t_stat,3),
            "df":        min_len - 1,
            "p":         round(t_p,4),
            "sig":       "***" if t_p<0.001 else "**" if t_p<0.01 else "*" if t_p<0.05 else "ns"
        }

    # ── Save Table 5.3 as text ────────────────────────────────────────────────
    tbl_path = os.path.join(OUTPUT_DIR, "correlation_results.txt")
    with open(tbl_path, "w") as f:
        f.write("TABLE 5.3 — Correlation Analysis Results\n")
        f.write("="*75 + "\n")
        f.write(f"{'Correlation Pair':<45} {'Pearson r':>9} {'p':>7} {'Spearman r':>10} {'Sig':>5} {'n':>4}\n")
        f.write("-"*75 + "\n")
        for r in results:
            pr = f"{r['pr']:.4f}" if isinstance(r['pr'],float) else r['pr']
            pp = f"{r['pp']:.4f}" if isinstance(r['pp'],float) else r['pp']
            sr = f"{r['sr']:.4f}" if isinstance(r['sr'],float) else r['sr']
            f.write(f"{r['label']:<45} {pr:>9} {pp:>7} {sr:>10} {r['sig']:>5} {r['n']:>4}\n")
        f.write("\n* p<0.05  ** p<0.01  *** p<0.001  ns=not significant\n")
        if ttest_result:
            f.write("\nPre vs Post Stress (paired t-test):\n")
            f.write(f"  Pre-session  : mean={ttest_result['pre_mean']}  SD={ttest_result['pre_std']}\n")
            f.write(f"  Post-session : mean={ttest_result['post_mean']}  SD={ttest_result['post_std']}\n")
            f.write(f"  t({ttest_result['df']}) = {ttest_result['t']}  p = {ttest_result['p']} {ttest_result['sig']}\n")
    print(f"  Table 5.3 saved → {tbl_path}")

    # Print Table 5.3 to console
    print("\n  TABLE 5.3 — Copy these values into your dissertation:\n")
    print(f"  {'Pair':<45} {'Pearson r':>9} {'p':>7} {'Spearman r':>10} {'Sig':>5}")
    print("  " + "-"*75)
    for r in results:
        pr = f"{r['pr']:.4f}" if isinstance(r['pr'],float) else r['pr']
        pp = f"{r['pp']:.4f}" if isinstance(r['pp'],float) else r['pp']
        sr = f"{r['sr']:.4f}" if isinstance(r['sr'],float) else r['sr']
        print(f"  {r['label']:<45} {pr:>9} {pp:>7} {sr:>10} {r['sig']:>5}")
    if ttest_result:
        print(f"\n  Pre-session stress  : {ttest_result['pre_mean']} ± {ttest_result['pre_std']}")
        print(f"  Post-session stress : {ttest_result['post_mean']} ± {ttest_result['post_std']}")
        print(f"  t({ttest_result['df']}) = {ttest_result['t']},  p = {ttest_result['p']} {ttest_result['sig']}")

    # ── Figure 5.3 — Scatter plot ─────────────────────────────────────────────
    print("\n[Generating] Figure 5.3 — Correlation Scatter Plot...")

    # Pick the most meaningful pair for Figure 5.3
    # (Detected Engaged % vs Self-reported Comprehension)
    main_result = results[0]
    xs = main_result["xs"]
    ys = main_result["ys"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Correlation Analysis — Detected States vs Self-reported Comprehension",
                 fontsize=13, fontweight="bold", y=1.02)

    plot_pairs = [
        (results[0], axes[0], "#6366f1", "Detected Engaged %", "Self-reported Comprehension (1–5)"),
        (results[1], axes[1], "#ef4444", "Detected Confused %", "Self-reported Comprehension (1–5)"),
    ]

    for res, ax, col, xlabel, ylabel in plot_pairs:
        ax.set_facecolor("white")
        xs_p = res["xs"]
        ys_p = res["ys"]

        if len(xs_p) >= 2:
            ax.scatter(xs_p, ys_p, color=col, s=100, alpha=0.8,
                       edgecolors="white", linewidth=1.2, zorder=5)
            # Regression line
            m, b = np.polyfit(xs_p, ys_p, 1)
            x_line = np.linspace(min(xs_p), max(xs_p), 100)
            ax.plot(x_line, m*x_line+b, color=col, linewidth=2,
                    linestyle="--", alpha=0.8, zorder=4)
            # Label each point
            for x_val, y_val in zip(xs_p, ys_p):
                ax.annotate(f"({x_val:.0f}%, {y_val:.0f})",
                            (x_val, y_val), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color="#555555")

        pr_str = f"r = {res['pr']:.3f}" if isinstance(res['pr'],float) else "r = N/A"
        pp_str = f"p = {res['pp']:.4f} {res['sig']}" if isinstance(res['pp'],float) else ""
        ax.set_title(f"{res['label']}\n{pr_str}  {pp_str}  n={res['n']}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "correlation_chart.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  FULL ANALYSIS REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(sus_result, corr_results):
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    lines = [
        "KIU Comprehension System — Full Statistical Analysis Report",
        "="*60,
        "",
        f"SUS Participants   : {len(sus_scores) if sus_scores else 0}",
        f"Session Readings   : {len(log_rows) if log_rows else 0}",
        "",
    ]

    if sus_result:
        mean, std, grade, n = sus_result
        lines += [
            "SUS SCORE SUMMARY",
            "-"*40,
            f"  n              : {n}",
            f"  Mean           : {mean:.1f}",
            f"  Std deviation  : {std:.1f}",
            f"  Min            : {min(sus_scores):.1f}",
            f"  Max            : {max(sus_scores):.1f}",
            f"  Grade          : {grade}",
            f"  Benchmark [3]  : 83.33 (Putra & Arifin, 2019)",
            "",
        ]

    if corr_results:
        lines += [
            "CORRELATION RESULTS (Table 5.3)",
            "-"*40,
        ]
        for r in corr_results:
            pr = f"{r['pr']:.4f}" if isinstance(r['pr'],float) else r['pr']
            pp = f"{r['pp']:.4f}" if isinstance(r['pp'],float) else r['pp']
            sr = f"{r['sr']:.4f}" if isinstance(r['sr'],float) else r['sr']
            lines.append(f"  {r['label']}")
            lines.append(f"    Pearson r={pr}  p={pp} {r['sig']}  Spearman r={sr}  n={r['n']}")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Full report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
sus_result  = generate_figure_52()
corr_result = generate_correlations_and_figure53()
generate_report(sus_result, corr_result)

print("\n" + "="*60)
print("  OUTPUT FILES (in results/ folder)")
print("="*60)
for fname in ["sus_score_distribution.png","correlation_chart.png",
              "correlation_results.txt","analysis_report.txt"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    status = "✓ created" if os.path.exists(fpath) else "✗ not generated"
    print(f"  {fname:<35} {status}")
print("\n  Use these files for your dissertation:")
print("  Figure 5.2 → sus_score_distribution.png")
print("  Figure 5.3 → correlation_chart.png")
print("  Table 5.3  → copy values from correlation_results.txt")
print("="*60 + "\n")
