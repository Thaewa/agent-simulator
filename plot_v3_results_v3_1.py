"""
plot_v3_results_v3_1.py
Visualization suite for wasp-larvae simulation with explicit color mapping by pathfinding mode.

Each chart now visually separates modes with fixed color codes:
- greedy: orange (#E69F00)
- pheromone: green (#009E73)
- random: blue (#0072B2)
- gradient: magenta (#CC79A7)

Output folder: ./output_plots_v3_1/
"""

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metrics import normalize_aggregate, summarize_runs, validate_schema

# ==============================================================
# SETUP
# ==============================================================

BASE = Path(".")
LOGS = BASE / "output_logs"
OUT = BASE / "output_plots_v3_1"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "greedy": "#E69F00",              # orange
    "random_walk": "#009E73",         # green
    "tsp": "#0072B2",                 # blue
    "random_walk-biased": "#CC79A7",  # magenta
    "tsp-hamiltonian": "#999999"      # gray tone for clarity
}

def get_color(mode: str):
    """
    Return color strictly based on mode, matching lowercase folder names.
    Raises an error if mode not in COLORS (helps catch typos).
    """
    key = mode.lower().strip()
    if key not in COLORS:
        raise ValueError(f"[ERROR] No color defined for mode '{mode}'")
    return COLORS[key]

def load_csvs(pattern):
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            mode = Path(f).parts[-2].lower() if "output_logs" in f else "unknown"
            df["__mode__"] = mode
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==============================================================
# PLOTS
# ==============================================================

def plot_caption(text="Each color corresponds to a distinct pathfinding algorithm."):
    plt.figtext(
        0.5, -0.04,
        text,
        ha="center", fontsize=8, color="gray"
    )

# 1️ Req 1 + 6: Larvae hunger over time (mean/min/max)
def plot_larvae_hunger_over_time():
    df = load_csvs("output_logs/*/larvae_log_*.csv")
    if df.empty:
        print("[SKIP] No larvae logs.")
        return

    modes = sorted(df["__mode__"].unique())
    n = len(modes)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]  # make iterable if only one mode

    for ax, mode in zip(axes, modes):
        grp = df[df["__mode__"] == mode]
        agg = grp.groupby("timestamp")["hunger_level"].agg(["mean", "min", "max"]).reset_index()

        color = get_color(mode)
        ax.plot(agg["timestamp"], agg["mean"], color=color, label=mode.capitalize())
        ax.fill_between(agg["timestamp"], agg["min"], agg["max"], color=color, alpha=0.2)

        ax.set_title(f"{mode.capitalize()} mode")
        ax.set_ylabel("Hunger level")
        ax.set_ylim([0.8, 3])  # set max y to 10
        ax.legend()

    axes[-1].set_xlabel("Time step")

    plt.suptitle("Larvae hunger over time (mean ± range) by algorithm", fontsize=12)
    plot_caption()
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(OUT / "1_larvae_hunger_over_time_per_mode.png", dpi=150, bbox_inches="tight")
    plt.close()

# 2️ Req 2: Larvae hunger vs distance
def plot_larvae_distance_vs_hunger():
    df = load_csvs("output_logs/*/larvae_log_*.csv")
    fig, ax = plt.subplots(2, 1, figsize=(12, 15), sharex=True)
    # fig = plt.figure(figsize=(12, 15))
    # fig, ax = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    import seaborn as sns
    grp = df.groupby(["__mode__", "simulation_id", "larva_id", "position_x", "position_y"]).agg(
            mean_hunger=("hunger_level", "mean"),
            min_hunger=("hunger_level", "min"),
            max_hunger=("hunger_level", "max"),
            dist=("distance_to_nest", "mean")).reset_index()
    # Binarize the 'dist' column into 5 equally spaced levels
    grp['dist_bin'] = pd.cut(grp['dist'], bins=5)

    sns.boxplot(data=grp, x="dist_bin", y="mean_hunger", hue="__mode__", ax=ax[0], palette=COLORS)
    sns.boxplot(data=grp, x="dist_bin", y="max_hunger", hue="__mode__", ax=ax[1],  palette=COLORS)
    for ax_ in ax:
        ax_.grid(axis="y", linestyle=":", color="gray", zorder=-1)
        ax_.legend(title=None)
    
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "2_larvae_distance_vs_hunger.png", dpi=150, bbox_inches="tight")
    plt.close()

# 3️ Req 3: Wasp hunger cue vs time
def plot_wasp_hunger_cue_over_time():
    df_agent = load_csvs("output_logs/*/agent_log_*.csv")
    df_nest = load_csvs("output_logs/*/larvae_log_*.csv")

    if df_agent.empty or "hunger_cue" not in df_agent.columns:
        print("[SKIP] No agent hunger cue.")
        return
    df_agent = df_agent[df_agent["agent_role"].astype(str).str.upper() == "FEEDER"]

    if df_nest.empty:
        print("[SKIP] No larvae logs.")
        return

    modes = sorted(df_nest["__mode__"].unique())
    n = len(modes)

    # Create the figure and subplots once
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]  # Handle single-mode case by wrapping axes in a list

    for i, mode in enumerate(modes):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        grp_nest = df_nest[df_nest["__mode__"] == mode]
        grp_agent = df_agent[df_agent["__mode__"] == mode]

        color = get_color(mode)

        agg_larvae = grp_nest.groupby("timestamp")["hunger_level"].mean().reset_index()
        agg_wasp = grp_agent.groupby("timestamp")["hunger_cue"].mean().reset_index()

        # Larvae hunger (solid line)
        ax1.plot(agg_larvae["timestamp"], agg_larvae["hunger_level"],
                 color=color, linewidth=2.5, label="Larvae hunger level")

        # Wasp hunger cue (dashed line)
        ax2.plot(agg_wasp["timestamp"], agg_wasp["hunger_cue"],
                 color=color, linestyle="--", alpha=0.6, label="Wasp hunger cue")

        # Set titles and axis labels
        ax1.set_title(f"{mode.capitalize()} mode")
        ax1.set_ylabel("Larvae Hunger Level", color=color)
        ax2.set_ylabel("Wasp Hunger Cue", color=color)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Display legends on separate sides
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time step")

    plt.suptitle("Wasp vs Larvae Hunger Over Time (Dual Scale per Mode)", fontsize=12)
    plot_caption()
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(OUT / "3_wasp_hunger_cue_over_time_all_modes.png", dpi=150, bbox_inches="tight")
    plt.close()



# 4️ Req 4: Forager count vs hunger cue
def plot_foragers_vs_hunger_cue():
    """
    Generate a box plot of hunger cue distribution per number of foragers,
    with data points colored by pathfinding mode.
    """
    n = load_csvs("output_logs/*/nest_log_*.csv")
    a = load_csvs("output_logs/*/agent_log_*.csv")
    if n.empty or a.empty:
        print("[SKIP] Missing logs for forager-hunger.")
        return

    a_mean = a.groupby(["__mode__","timestamp"])["hunger_cue"].mean().reset_index()
    merged = n.merge(a_mean, on=["__mode__","timestamp"], how="inner")

    plt.figure(figsize=(10, 6))
    # Use seaborn for a categorical plot (boxplot)
    import seaborn as sns
    sns.boxplot(data=merged, x="total_foragers", y="hunger_cue", hue="__mode__", palette=COLORS)

    plt.xlabel("# Foragers")
    plt.ylabel("Average hunger cue")
    plt.title("Hunger Cue Distribution by Number of Foragers")
    plt.legend()
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "4_foragers_vs_hunger_cue.png", dpi=150, bbox_inches="tight")
    plt.close()

# 5️ Req 5: Hunger overlay across algorithms
def plot_hunger_multi_pathfinding():
    df = load_csvs("output_logs/*/larvae_log_*.csv")
    if df.empty:
        print("[SKIP] No larvae logs for overlay.")
        return
    plt.figure(figsize=(7,4))
    for mode, grp in df.groupby("__mode__"):
        agg = grp.groupby("timestamp")["hunger_level"].mean().reset_index()
        plt.plot(agg["timestamp"], agg["hunger_level"],
                 label=mode, color=COLORS.get(mode,"gray"))
    plt.xlabel("Time step")
    plt.ylabel("Average larvae hunger")
    plt.title("Larvae hunger over time across algorithms")
    plt.legend()
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "6_larvae_hunger_multi_pathfinding.png", dpi=150, bbox_inches="tight")
    plt.close()

def run_errors(series):
    """Omit uncertainty when there are no repeated runs."""
    return series if series.notna().any() else None


def plot_foraging_efficiency():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    if df.empty:
        return
    columns = ["mean_distance_per_feed_larvae", "mean_distance_per_feed_wasp",
               "final_low_hunger_fraction"]
    agg = summarize_runs(df, columns)
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    for offset, column, label, color in (
        (-width/2, columns[0], "Larval feeding event", "#56B4E9"),
        (width/2, columns[1], "Adult transfer event", "#E69F00"),
    ):
        ax.bar(x + offset, agg[column + "_mean"], width,
               yerr=run_errors(agg[column + "_std"]), label=label, color=color, capsize=4)
    ax.set_xticks(x, agg["__mode__"], rotation=15)
    ax.set(ylabel="Total adult distance / successful event",
           title="Adult travel cost per successful feeding event")
    ax.legend()
    plot_caption("Colors identify event types; missing costs mean no corresponding events.")
    plt.tight_layout()
    plt.savefig(OUT / "7_foraging_efficiency_compare.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    column = "final_low_hunger_fraction"
    ax.bar(agg["__mode__"], agg[column + "_mean"],
           yerr=run_errors(agg[column + "_std"]),
           color=[get_color(mode) for mode in agg["__mode__"]], capsize=4)
    ax.set(ylabel="Fraction of larvae with final hunger < 1.5",
           title="Final low-hunger fraction (a state measure)", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7b_feeding_efficiency_by_algorithm.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_foraging_frequency():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    if df.empty:
        return
    columns = ["mean_feed_rate_wasp", "mean_feed_rate_larvae"]
    agg = summarize_runs(df, columns)
    x, width = np.arange(len(agg)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, column, label, color in (
        (-width/2, columns[0], "Adult transfers", "#E69F00"),
        (width/2, columns[1], "Larval feeds", "#56B4E9"),
    ):
        ax.bar(x + offset, agg[column + "_mean"], width,
               yerr=run_errors(agg[column + "_std"]), label=label, color=color, capsize=4)
    ax.set_xticks(x, agg["__mode__"], rotation=15)
    ax.set(ylabel="Successful events per adult per step", title="Feeding event rates")
    ax.legend()
    plot_caption("Colors identify event types.")
    plt.figtext(0.5, -0.09, "Error bars: between-run SD; omitted when fewer than 2 runs.",
                ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(OUT / "8_foraging_frequency_compare.png", dpi=150, bbox_inches="tight")
    plt.close()


# 7c: Total distance traveled (feeders) vs total larvae fed
def plot_total_distance_vs_feed():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    df = normalize_aggregate(df)
    if df.empty:
        print("[SKIP] No aggregate data for total distance vs feed.")
        return

    plt.figure(figsize=(6, 5))
    for mode, grp in df.groupby("__mode__"):
        color = COLORS.get(mode, "gray")
        plt.scatter(grp["distance_per_wasp_per_step"], grp["feed_per_larva"],
                    s=60, color=color, alpha=0.7, label=mode.capitalize())

    plt.xlabel("Distance per feeder-role step")
    plt.ylabel("Feeder larval feeds per larva")
    plt.title("7c. Normalized: Feeder distance per step vs larval feeding")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Pathfinding Mode")
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7c_total_distance_vs_feed.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7c1: Total distance vs feed (mean ± std across simulations)
def plot_total_distance_vs_feed_avg():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    df = normalize_aggregate(df)
    if df.empty:
        print("[SKIP] No aggregate data for 7c1.")
        return

    agg = df.groupby(["__mode__", "simulation_id"], as_index=False).agg({
        "distance_per_wasp_per_step": "mean",
        "feed_per_larva": "mean"
    })

    mean_df = agg.groupby("__mode__", as_index=False).agg(
        mean_x=("distance_per_wasp_per_step", "mean"),
        std_x=("distance_per_wasp_per_step", "std"),
        mean_y=("feed_per_larva", "mean"),
        std_y=("feed_per_larva", "std")
    )

    plt.figure(figsize=(6,5))
    for _, row in mean_df.iterrows():
        color = COLORS.get(row["__mode__"], "gray")
        plt.errorbar(row["mean_x"], row["mean_y"],
                     xerr=row["std_x"] if pd.notna(row["std_x"]) else None,
                     yerr=row["std_y"] if pd.notna(row["std_y"]) else None,
                     fmt='o', color=color, label=row["__mode__"].capitalize(),
                     capsize=4)
    plt.xlabel("Distance per feeder-role step")
    plt.ylabel("Feeder larval feeds per larva")
    plt.title("7c1. Averaged: Feeder distance per step vs larval feeding")
    plt.legend(title="Pathfinding Mode")
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7c1_total_distance_vs_feed_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7d: Walk-to-Feed Efficiency (Larvae fed per distance)
def plot_walk_to_feed_efficiency():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    df = normalize_aggregate(df)
    if df.empty:
        print("[SKIP] No aggregate data for 7d efficiency plot.")
        return

    plt.figure(figsize=(6, 5))
    for mode, grp in df.groupby("__mode__"):
        color = COLORS.get(mode, "gray")
        plt.scatter(grp["distance_per_wasp_per_step"], grp["norm_walk_to_feed_eff"],
                    s=60, color=color, alpha=0.7, label=mode.capitalize())

    plt.xlabel("Distance per feeder-role step")
    plt.ylabel("Feeder larval feeds per distance unit")
    plt.title("7d. Normalized: Feeder larval feeds per distance unit")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Pathfinding Mode")
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7d_walk_to_feed_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7d1: Walk-to-Feed Efficiency (mean ± std across simulations)
def plot_walk_to_feed_efficiency_avg():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    df = normalize_aggregate(df)
    if df.empty:
        print("[SKIP] No aggregate data for 7d1.")
        return

    agg = df.groupby(["__mode__", "simulation_id"], as_index=False).agg({
        "distance_per_wasp_per_step": "mean",
        "norm_walk_to_feed_eff": "mean"
    })

    mean_df = agg.groupby("__mode__", as_index=False).agg(
        mean_x=("distance_per_wasp_per_step", "mean"),
        std_x=("distance_per_wasp_per_step", "std"),
        mean_y=("norm_walk_to_feed_eff", "mean"),
        std_y=("norm_walk_to_feed_eff", "std")
    )

    plt.figure(figsize=(6,5))
    for _, row in mean_df.iterrows():
        color = COLORS.get(row["__mode__"], "gray")
        plt.errorbar(row["mean_x"], row["mean_y"],
                     xerr=row["std_x"] if pd.notna(row["std_x"]) else None,
                     yerr=row["std_y"] if pd.notna(row["std_y"]) else None,
                     fmt='o', color=color, label=row["__mode__"].capitalize(),
                     capsize=4)
    plt.xlabel("Distance per feeder-role step")
    plt.ylabel("Feeder larval feeds per distance unit")
    plt.title("7d1. Averaged: Feeder larval feeds per distance unit")
    plt.legend(title="Pathfinding Mode")
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7d1_walk_to_feed_efficiency_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7e: Total Steps vs Total Feed Count
def plot_total_steps_vs_feed():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    df = normalize_aggregate(df)
    if df.empty:
        print("[SKIP] No aggregate data for total steps vs feed.")
        return

    plt.figure(figsize=(6, 5))
    for mode, grp in df.groupby("__mode__"):
        color = COLORS.get(mode, "gray")
        plt.scatter(grp["norm_steps"], grp["feed_per_larva"],
                    s=60, color=color, alpha=0.7, label=mode.capitalize())

    plt.xlabel("Feeder-role steps per participating adult")
    plt.ylabel("Feeder larval feeds per larva")
    plt.title("7e. Normalized: Feeder exposure vs larval feeding")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Pathfinding Mode")
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7e_total_steps_vs_feed.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7e1: Steps vs Feeding Count (mean ± std across simulations)
def plot_total_steps_vs_feed_avg():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    df = normalize_aggregate(df)
    if df.empty:
        print("[SKIP] No aggregate data for 7e1.")
        return

    agg = df.groupby(["__mode__", "simulation_id"], as_index=False).agg({
        "norm_steps": "mean",
        "feed_per_larva": "mean"
    })

    mean_df = agg.groupby("__mode__", as_index=False).agg(
        mean_x=("norm_steps", "mean"),
        std_x=("norm_steps", "std"),
        mean_y=("feed_per_larva", "mean"),
        std_y=("feed_per_larva", "std")
    )

    plt.figure(figsize=(6,5))
    for _, row in mean_df.iterrows():
        color = COLORS.get(row["__mode__"], "gray")
        plt.errorbar(row["mean_x"], row["mean_y"],
                     xerr=row["std_x"] if pd.notna(row["std_x"]) else None,
                     yerr=row["std_y"] if pd.notna(row["std_y"]) else None,
                     fmt='o', color=color, label=row["__mode__"].capitalize(),
                     capsize=4)
    plt.xlabel("Feeder-role steps per participating adult")
    plt.ylabel("Feeder larval feeds per larva")
    plt.title("7e1. Averaged: Feeder exposure vs larval feeding")
    plt.legend(title="Pathfinding Mode")
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7e1_total_steps_vs_feed_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

# 8️ Req 10: Wasp trajectory (x,y)
def plot_wasp_trajectory():
    df = load_csvs("output_logs/*/agent_log_*.csv")
    if df.empty or "position_x" not in df.columns:
        print("[SKIP] No agent trajectories.")
        return
    df = df[df["agent_id"].astype(str).str.startswith("W")]
    for mode, grp in df.groupby("__mode__"):
        plt.figure(figsize=(6,5))
        wasps = list(grp["agent_id"].unique())[:10]
#        active_counts = grp.groupby("agent_id")["position_x"].nunique().sort_values(ascending=False)
#        wasps = list(active_counts.head(10).index)

        for wid in wasps:
            seg = grp[grp["agent_id"]==wid].sort_values("timestamp")
            plt.plot(seg["position_x"], seg["position_y"], label=wid)
        plt.title(f"Wasp trajectories ({mode})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(fontsize="x-small")
        plot_caption()
        plt.tight_layout()
        plt.savefig(OUT / f"9_wasp_trajectory_{mode}.png", dpi=150, bbox_inches="tight")
        plt.close()

# ==============================================================
# MAIN
# ==============================================================

def main():
    
    aggregates = load_csvs("output_logs/*/aggregate_results_*.csv")
    if not aggregates.empty:
        validate_schema(aggregates)
    print(f"[INFO] Saving charts into: {OUT.resolve()}")
    plot_larvae_hunger_over_time()
    plot_larvae_distance_vs_hunger()
    plot_wasp_hunger_cue_over_time()
    plot_foragers_vs_hunger_cue()
    plot_hunger_multi_pathfinding()
    plot_foraging_efficiency()
    plot_total_distance_vs_feed()   # <---- 7c
    plot_total_distance_vs_feed_avg()  # 7c1
    plot_walk_to_feed_efficiency()  # <---- 7d
    plot_walk_to_feed_efficiency_avg() # 7d1
    plot_total_steps_vs_feed()      # <---- 7e
    plot_total_steps_vs_feed_avg()     # 7e1
    plot_foraging_frequency()
    plot_wasp_trajectory()
    print("[INFO] Done. All charts ready for Capstone visualization showcase.")

if __name__ == "__main__":
    main()
