# gen_chart.py — Matplotlib-only charts with Baseline support and overlap handling
import os, sys, json
import numpy as np
import pandas as pd

# --- Matplotlib headless setup ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ================== Defaults & Paths ==================
DEFAULT_LOG_PATH = r"C:\Users\Enes\Desktop\cyber\results\full_experiment_log.json"
DEFAULT_OUT_DIR  = r"C:\Users\Enes\Desktop\cyber\out"

# ================== Style / constants ==================
COLORS = {
    "PPO":      "#E41A1C",  # red
    "SAC":      "#377EB8",  # blue
    "DQN":      "#4DAF4A",  # green
    "A2C":      "#FFD700",  # yellow
    "TD3":      "#FF7F00",  # orange
    "Baseline": "#FFC0CB",  # pink
}
MARKERS = {
    "PPO": "o", "SAC": "s", "DQN": "D",
    "A2C": "^", "TD3": "x", "Baseline": "P"
}
ALGORITHM_ORDER = ["PPO", "SAC", "DQN", "A2C", "TD3", "Baseline"]
PHASE_ORDER = ["Early Phase", "Mid Phase", "Extended Phase", "Pre-Final Phase", "Final Phase"]

TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE  = 9
LEGEND_KW = dict(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)

# ================== Data loading / prep ==================
def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_algorithm_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    mapping = {
        "ppo":"PPO", "sac":"SAC", "dqn":"DQN", "a2c":"A2C", "td3":"TD3",
        "baseline":"Baseline", "rule_based":"Baseline", "rule-based":"Baseline"
    }
    return mapping.get(key, name.strip())

def tidy_df(data):
    rows = []

    # 1) RL algorithms
    for algorithm, phases in data.get("algorithms", {}).items():
        for phase_key, rec in (phases or {}).items():
            rec = rec or {}
            rows.append({
                "Algorithm": _normalize_algorithm_name(algorithm),
                "Phase": rec.get("phase", phase_key),
                "train_episodes": rec.get("train_episodes"),
                "test_episodes": rec.get("test_episodes"),
                "timesteps_per_train_episode": rec.get("timesteps_per_train_episode"),
                "total_timesteps": rec.get("total_timesteps"),
                "training_time_sec": rec.get("training_time_sec"),
                "fps": rec.get("fps"),
                "iterations_est": rec.get("iterations_est"),
                "eval_reward_mean": rec.get("eval_reward_mean"),
                "eval_reward_std": rec.get("eval_reward_std"),
                "eval_reward_min": rec.get("eval_reward_min"),
                "eval_reward_max": rec.get("eval_reward_max"),
            })

    # 2) Baseline (test-only; no training/fps per your JSON)
    for phase_key, rec in (data.get("baseline", {}) or {}).items():
        rec = rec or {}
        rows.append({
            "Algorithm": "Baseline",
            "Phase": rec.get("phase", phase_key),
            "train_episodes": None,
            "test_episodes": rec.get("test_episodes"),
            "timesteps_per_train_episode": None,
            "total_timesteps": None,
            "training_time_sec": None,
            "fps": None,
            "iterations_est": None,
            "eval_reward_mean": rec.get("eval_reward_mean"),
            "eval_reward_std":  rec.get("eval_reward_std"),
            "eval_reward_min":  rec.get("eval_reward_min"),
            "eval_reward_max":  rec.get("eval_reward_max"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Log parsed but no rows found.")

    present = [p for p in PHASE_ORDER if p in set(df["Phase"].dropna().tolist())]
    if present:
        df["Phase"] = pd.Categorical(df["Phase"], categories=present, ordered=True)
        df = df.sort_values(["Algorithm", "Phase"])
    return df

def compute_overall_winner(df):
    metrics = ["eval_reward_mean", "fps"]  # higher is better
    per_algorithm = df.groupby("Algorithm")[metrics].mean(numeric_only=True)
    zsum = {}
    for metric in metrics:
        vals = per_algorithm[metric].astype(float)
        # neutralize missing metrics (e.g., Baseline has no fps)
        vals = vals.fillna(vals.mean())
        std = vals.std(ddof=0)
        if std == 0 or np.isnan(std):
            z = (vals - vals.mean())
        else:
            z = (vals - vals.mean()) / (std + 1e-9)
        z = z.fillna(0.0)
        for algorithm, zval in z.items():
            zsum[algorithm] = zsum.get(algorithm, 0.0) + float(zval)
    ranked = sorted(zsum.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, metrics

# ================== Utils: save, theme, overlap handling ==================
def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

def _line_stroke():
    # white stroke behind lines/markers to keep them visible when overlapping
    return [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]

def resolve_overlap_by_phase(phases, values_by_alg, tol=1e-9):
    """
    For each phase index, detect algorithms whose values are equal (within tol).
    Return dict: alg -> jittered_y (np.array) so lines/markers don't overlap.
    Jitter is small relative to data range and symmetric around the original value.
    """
    algs = list(values_by_alg.keys())
    if not algs:
        return {}
    n = len(phases)
    Y = np.vstack([np.asarray(values_by_alg[a], dtype=float) for a in algs])  # shape (A, n)

    # Determine scale per column for jitter size
    with np.errstate(all="ignore"):
        col_min = np.nanmin(Y, axis=0)
        col_max = np.nanmax(Y, axis=0)
    span = col_max - col_min
    span[~np.isfinite(span)] = 1.0  # if all-nan or inf -> safe default
    base_delta = np.maximum(span * 0.01, 1.0)  # 1% of span (or 1 unit if span≈0)

    # For each column, group equal values and assign jitters
    Yj = Y.copy()
    for j in range(n):
        col = Y[:, j]
        if np.all(np.isnan(col)):  # nothing to plot
            continue
        groups = {}  # value_key -> list of alg indices
        for i, v in enumerate(col):
            if np.isnan(v):
                continue
            matched_key = None
            for key in list(groups.keys()):
                if abs(v - key) <= tol:
                    matched_key = key
                    break
            if matched_key is None:
                groups[v] = [i]
            else:
                groups[matched_key].append(i)

        # Apply symmetric jitter within each group of size>1
        for _, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            k = len(idxs)
            offsets = np.linspace(-(k-1)/2, (k-1)/2, k)
            Yj[idxs, j] = col[idxs] + offsets * base_delta[j]

    return {a: Yj[ai, :] for ai, a in enumerate(algs)}

# ================== Matplotlib charts ==================
def charts_mpl(df, outdir):
    from pandas.api.types import CategoricalDtype

    present_set = set(df["Algorithm"].unique())
    algs_all = [a for a in ALGORITHM_ORDER if a in present_set] + \
               sorted([a for a in present_set if a not in ALGORITHM_ORDER])

    if isinstance(df["Phase"].dtype, CategoricalDtype):
        phases = list(df["Phase"].cat.categories)
    else:
        phases = sorted(df["Phase"].dropna().unique().tolist())
        df["Phase"] = pd.Categorical(df["Phase"], categories=phases, ordered=True)
        df = df.sort_values(["Algorithm", "Phase"])

    x_idx = np.arange(len(phases))

    # helper: select only algorithms with at least one finite value for a given metric
    def _algs_with_metric(metric):
        ok = []
        for a in algs_all:
            vals = df.loc[df["Algorithm"]==a, metric]
            if vals.notna().any():
                ok.append(a)
        return ok

    # helper: nice line plot with overlap resolution
    def line_by_phase(metric, title, filename, ylabel):
        algs = _algs_with_metric(metric)
        if not algs:
            return
        # collect raw series per algorithm aligned to phases
        values_by_alg = {}
        for a in algs:
            d = df[df["Algorithm"] == a]
            arr = []
            for ph in phases:
                v = d.loc[d["Phase"]==ph, metric]
                arr.append(float(v.iloc[0]) if not v.empty and pd.notna(v.iloc[0]) else np.nan)
            # only keep if not all nan
            if np.isfinite(arr).any():
                values_by_alg[a] = arr
        if not values_by_alg:
            return

        # resolve overlaps column-wise
        jittered = resolve_overlap_by_phase(phases, values_by_alg, tol=1e-9)

        fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
        for a in algs:
            if a not in jittered:
                continue
            y = jittered[a]
            ax.plot(
                x_idx, y,
                marker=MARKERS.get(a, "o"), linewidth=2, label=a,
                color=COLORS.get(a, "#444"), zorder=2,
                path_effects=_line_stroke()
            )
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Phase", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.legend(**LEGEND_KW)
        savefig(fig, outdir, filename)

    # 1–5: Reward + FPS (Baseline appears only where metric exists)
    line_by_phase("eval_reward_mean","Reward Mean by Phase","01_reward_mean_by_phase","Reward Mean")
    line_by_phase("eval_reward_std","Reward Std by Phase","02_reward_std_by_phase","Reward Std")
    line_by_phase("eval_reward_min","Reward Min by Phase","03_reward_min_by_phase","Reward Min")
    line_by_phase("eval_reward_max","Reward Max by Phase","04_reward_max_by_phase","Reward Max")
    line_by_phase("fps","Throughput (FPS) by Phase","05_fps_by_phase","Frames per Second")

    # 6: Training Time per Phase (grouped bars) — skip algorithms with all-NaN time (Baseline)
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
    algs_t = [a for a in algs_all if df.loc[df["Algorithm"]==a, "training_time_sec"].notna().any()]
    if algs_t:
        width = 0.8 / max(1, len(algs_t))
        for i, a in enumerate(algs_t):
            d = df[df["Algorithm"] == a]
            y = []
            for ph in phases:
                v = d.loc[d["Phase"]==ph, "training_time_sec"]
                y.append(float(v.iloc[0])/60.0 if not v.empty and pd.notna(v.iloc[0]) else np.nan)
            ax.bar(x_idx + i * width, y, width=width, label=a, color=COLORS.get(a, "#444"), zorder=2)
        ax.set_title("Training Time per Phase", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Phase", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Minutes", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(x_idx + (len(algs_t)-1)*width/2 if len(algs_t)>0 else x_idx)
        ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        ax.legend(**LEGEND_KW)
        savefig(fig, outdir, "06_training_time_by_phase")

    # generic bar-per-algorithm (drop NaN)
    def bar_per_algorithm(series, title, filename, ylabel):
        s = series.reindex(algs_all).dropna()
        if s.empty:
            return
        fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
        bars = ax.bar(range(len(s.index)), s.values.astype(float),
                      color=[COLORS.get(a, "#444") for a in s.index], zorder=2)
        for b in bars:
            b.set_edgecolor("white"); b.set_linewidth(0.5)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Algorithm", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(range(len(s.index)))
        ax.set_xticklabels(s.index, rotation=0, fontsize=TICK_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        savefig(fig, outdir, filename)

    # 7–12
    totals = df.groupby("Algorithm")["training_time_sec"].sum(numeric_only=True)
    bar_per_algorithm(totals/3600.0, "Total Training Time by Algorithm", "07_training_time_total_by_algorithm", "Hours")

    avg_reward = df.groupby("Algorithm")["eval_reward_mean"].mean(numeric_only=True)
    bar_per_algorithm(avg_reward, "Average Reward Across All Phases", "08_avg_reward_by_algorithm", "Reward Mean")

    peak_reward = df.groupby("Algorithm")["eval_reward_max"].max(numeric_only=True)
    bar_per_algorithm(peak_reward, "Peak Reward Achieved", "09_peak_reward_by_algorithm", "Reward Max")

    min_reward = df.groupby("Algorithm")["eval_reward_min"].min(numeric_only=True)
    bar_per_algorithm(min_reward, "Worst-Case Reward", "10_min_reward_by_algorithm", "Reward Min")

    avg_std = df.groupby("Algorithm")["eval_reward_std"].mean(numeric_only=True)
    bar_per_algorithm(avg_std, "Average Reward Variability", "11_avg_std_by_algorithm", "Std Dev")

    eff = (df["eval_reward_mean"] / (df["training_time_sec"]/3600.0).replace(0, np.nan))
    df_eff = df.assign(eff=eff).groupby("Algorithm")["eff"].mean(numeric_only=True)
    bar_per_algorithm(df_eff, "Efficiency: Reward per Training Hour", "12_eff_reward_per_hour", "Reward / hour")

    # 13: Total Timesteps by Phase (line)
    line_by_phase("total_timesteps","Total Timesteps by Phase","13_total_timesteps_by_phase","Timesteps")

    # 14: Iterations Estimate by Phase (line; include alg only if it has any non-NaN)
    def line_iterations():
        algs = [a for a in algs_all if df.loc[df["Algorithm"]==a, "iterations_est"].notna().any()]
        if not algs:
            return
        values_by_alg = {}
        for a in algs:
            d = df[df["Algorithm"]==a]
            arr = []
            for ph in phases:
                v = d.loc[d["Phase"]==ph, "iterations_est"]
                if v.empty or pd.isna(v.iloc[0]):
                    arr.append(np.nan)
                else:
                    arr.append(float(v.iloc[0]))
            if np.isfinite(arr).any():
                values_by_alg[a] = arr
        if not values_by_alg:
            return
        jittered = resolve_overlap_by_phase(phases, values_by_alg, tol=1e-9)
        fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
        for a in algs:
            if a not in jittered: continue
            y = jittered[a]
            ax.plot(x_idx, y, marker=MARKERS.get(a,"o"), linewidth=2, label=a,
                    color=COLORS.get(a,"#444"), zorder=2, path_effects=_line_stroke())
        ax.set_title("Iterations Estimate by Phase", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Phase", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Iterations (est.)", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
        ax.grid(True, alpha=0.25, zorder=0)
        ax.legend(**LEGEND_KW)
        savefig(fig, outdir, "14_iterations_est_by_phase")
    line_iterations()

    # 15: Phase winners (bars aligned to phases) — reward mean
    winners = []
    for ph in phases:
        sub = df[df["Phase"] == ph]
        # choose algorithm with max reward mean for this phase (ignoring NaN)
        if sub["eval_reward_mean"].notna().any():
            best_row = sub.loc[sub["eval_reward_mean"].idxmax()]
            winners.append((ph, best_row["Algorithm"], float(best_row["eval_reward_mean"])))
    if winners:
        phase_winner_df = pd.DataFrame(winners, columns=["Phase","Algorithm","score"])
        fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
        algs_w = [a for a in algs_all if a in phase_winner_df["Algorithm"].unique()]
        width = 0.8 / max(1, len(algs_w))
        for i, a in enumerate(algs_w):
            d = phase_winner_df[phase_winner_df["Algorithm"] == a]
            y = [np.nan]*len(phases)
            if not d.empty:
                pos = {ph:sc for ph,sc in zip(d["Phase"], d["score"])}
                for j, ph in enumerate(phases):
                    y[j] = pos.get(ph, np.nan)
            ax.bar(x_idx + i*width, y, width=width, label=a, color=COLORS.get(a, "#444"), zorder=2)
        ax.set_title("Phase Winners (by Reward Mean)", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Phase", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Reward Mean", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(x_idx + (len(algs_w)-1)*width/2 if len(algs_w)>0 else x_idx)
        ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        ax.legend(**LEGEND_KW)
        savefig(fig, outdir, "15_phase_winners_by_reward")

    # 16: Overall winner (z-sum)
    ranked, used_metrics = compute_overall_winner(df)
    if ranked:
        labels = [r[0] for r in ranked]
        vals = [r[1] for r in ranked]
        fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=150)
        colors = [COLORS.get(a, "#444") for a in labels]
        bars = ax.bar(labels, vals, color=colors, zorder=2)
        for b in bars:
            b.set_edgecolor("white"); b.set_linewidth(0.5)
        ax.set_title(f"Overall Winner (z-score sum of {', '.join(used_metrics)})", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("Algorithm", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Composite Score (higher is better)", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        savefig(fig, outdir, "16_overall_winner")

    # 17: Stability range (Reward Max - Reward Min)
    rng = (df.groupby("Algorithm")["eval_reward_max"].max(numeric_only=True)
         - df.groupby("Algorithm")["eval_reward_min"].min(numeric_only=True))
    bar_per_algorithm(rng, "Stability: Reward Range Across Phases", "17_reward_range_by_algorithm", "Reward Range (max−min)")

    # 18: Scatter: Reward vs FPS (skip algorithms with no FPS; Baseline has none)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=150)
    plotted = False
    for a in algs_all:
        d = df[df["Algorithm"]==a][["fps","eval_reward_mean"]].dropna()
        if d.empty:
            continue
        plotted = True
        ax.plot(d["fps"].astype(float).values, d["eval_reward_mean"].astype(float).values,
                "-o", label=a, color=COLORS.get(a,"#444"), path_effects=_line_stroke())
    if plotted:
        ax.set_title("Reward vs Throughput", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("FPS", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Reward Mean", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.25)
        ax.legend(**LEGEND_KW)
        savefig(fig, outdir, "18_scatter_reward_vs_fps")
    else:
        plt.close(fig)

    # 19: Heatmap Phase x Algorithm (Reward Mean) — include Baseline
    mat = df.pivot_table(index="Phase", columns="Algorithm",
                         values="eval_reward_mean", aggfunc="mean", observed=False)
    if not mat.empty:
        ordered_cols = [c for c in ALGORITHM_ORDER if c in mat.columns] + \
                       [c for c in mat.columns if c not in ALGORITHM_ORDER]
        mat = mat[ordered_cols]
        fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
        im = ax.imshow(mat.values, aspect="auto", origin="upper")
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=20, ha="right", fontsize=TICK_FONT_SIZE)
        ax.set_yticks(range(len(mat.index)));   ax.set_yticklabels(mat.index, fontsize=TICK_FONT_SIZE)
        ax.set_title("Reward Mean Matrix (Phase × Algorithm)", fontsize=TITLE_FONT_SIZE)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Reward Mean")
        savefig(fig, outdir, "19_heatmap_phase_algorithm_reward_mean")
    else:
        plt.close("all")

    # 20: 2D density (hist2d) FPS vs Reward — drop NaNs first
    d2 = df[["fps","eval_reward_mean"]].dropna()
    if not d2.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
        x = d2["fps"].astype(float).values
        y = d2["eval_reward_mean"].astype(float).values
        h = ax.hist2d(x, y, bins=24, cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="Count")
        ax.set_title("2D Density of Throughput vs Reward (All Algorithms)", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("FPS", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Reward Mean", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.15)
        savefig(fig, outdir, "20_density_heatmap_fps_vs_reward")
    else:
        plt.close("all")

    # 21: Correlation heatmap
    metrics_cols = ["eval_reward_mean","eval_reward_std","eval_reward_min","eval_reward_max",
                    "fps","training_time_sec","total_timesteps","iterations_est"]
    corr_df = df[metrics_cols].astype(float)
    if corr_df.dropna(how="all").shape[1] > 1:
        corr = corr_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=150)
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
        ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index, fontsize=TICK_FONT_SIZE)
        ax.set_title("Correlation Heatmap of Metrics", fontsize=TITLE_FONT_SIZE)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Correlation")
        savefig(fig, outdir, "21_correlation_heatmap_metrics")
    else:
        plt.close("all")

    # 22: Radar (spider) normalized — include Baseline where reward data exists
    algs_r = [a for a in algs_all if df.loc[df["Algorithm"]==a, "eval_reward_mean"].notna().any()]
    if algs_r:
        df_eff = df.assign(eff = df["eval_reward_mean"] / (df["training_time_sec"]/3600.0).replace(0, np.nan))
        agg = df_eff.groupby("Algorithm").agg({
            "eval_reward_mean": "mean",
            "eval_reward_std": "mean",
            "eval_reward_max": "max",
            "fps": "mean",
            "eff": "mean"
        }).reindex(algs_r)

        def _norm(s, invert=False):
            s = s.astype(float)
            # put missing values at the minimum to avoid unfair advantage
            s = s.fillna(s.min())
            if s.max()==s.min() or s.isna().all():
                n = pd.Series(0.5, index=s.index)
            else:
                n = (s - s.min()) / (s.max() - s.min())
            return (1-n) if invert else n

        radar_df = pd.DataFrame({
            "Reward Mean": _norm(agg["eval_reward_mean"]),
            "Reward Max": _norm(agg["eval_reward_max"]),
            "Throughput (FPS)": _norm(agg["fps"]),
            "Efficiency (Reward/hr)": _norm(agg["eff"]),
            "Stability (−Std)": _norm(agg["eval_reward_std"], invert=True),
        })
        cats = list(radar_df.columns)
        angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=150, subplot_kw=dict(polar=True))
        for a in algs_r:
            vals = radar_df.loc[a, cats].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, label=a, color=COLORS.get(a,"#444"), path_effects=_line_stroke())
            ax.fill(angles, vals, alpha=0.10, color=COLORS.get(a,"#444"))
        ax.set_title("Composite Performance (Normalized Radar)", fontsize=TITLE_FONT_SIZE, pad=20)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=TICK_FONT_SIZE)
        ax.set_yticklabels([]); ax.set_ylim(0,1)
        ax.legend(**LEGEND_KW)
        savefig(fig, outdir, "22_radar_composite_normalized")

def main():
    if len(sys.argv) == 1:
        log_path = DEFAULT_LOG_PATH
        out_dir  = DEFAULT_OUT_DIR
        print(f"[info] Using defaults:\n  log: {log_path}\n  out: {out_dir}")
    elif len(sys.argv) == 3:
        log_path = sys.argv[1]
        out_dir  = sys.argv[2]
    else:
        print("Usage: python gen_chart.py [<path_to_full_experiment_log.json> <output_dir>]")
        sys.exit(1)

    data = load_log(log_path)
    df = tidy_df(data)

    from pandas.api.types import CategoricalDtype
    alg_list = sorted(df['Algorithm'].unique().tolist())
    print("\nAlgorithms found:", alg_list)
    phases_list = df['Phase'].cat.categories.tolist() if isinstance(df['Phase'].dtype, CategoricalDtype) \
                   else sorted(df['Phase'].dropna().unique().tolist())
    print("Phases found:", phases_list)

    charts_mpl(df, out_dir)

    # Winner summary
    ranked, metrics = compute_overall_winner(df)
    lines = ["Overall ranking (higher is better) based on z-scores of: " + ", ".join(metrics)]
    for i, (algorithm, score) in enumerate(ranked, 1):
        lines.append(f"{i}. {algorithm}: {score:.3f}")
    summary_path = os.path.join(out_dir, "overall_winner_summary.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {summary_path}")
    print("[done] Charts generated")

if __name__ == "__main__":
    main()
