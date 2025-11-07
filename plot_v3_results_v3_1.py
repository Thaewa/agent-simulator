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

def normalize_aggregate(df):
    """
    Normalize relevant metrics per number of larvae and wasps.
    Adds unified normalization fields used by all 7x plots.
    """
    if df.empty:
        return df

    required = {"sum_distance_traveled_feeders", "sum_feed_count_larvae", "sum_step_count_feeders", "num_wasps", "num_larvae"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] normalize_aggregate: Missing columns {missing}")
        return df

    df = df.copy()
    df["num_wasps"] = df["num_wasps"].replace(0, np.nan)
    df["num_larvae"] = df["num_larvae"].replace(0, np.nan)
    df["sum_step_count_feeders"] = df["sum_step_count_feeders"].replace(0, np.nan)

    # --- Normalized base values ---
    df["norm_sum_distance"] = df["sum_distance_traveled_feeders"] / df["num_wasps"]
    df["norm_feed_count"] = df["sum_feed_count_larvae"] / df["num_larvae"]
    df["norm_steps"] = df["sum_step_count_feeders"] / df["num_wasps"]

    # --- Unified metrics (per wasp per step vs per larva) ---
    df["distance_per_wasp_per_step"] = df["sum_distance_traveled_feeders"] / (df["num_wasps"] * df["sum_step_count_feeders"])
    df["feed_per_larva"] = df["sum_feed_count_larvae"] / df["num_larvae"]

    # --- Efficiency ratios ---
    df["norm_walk_to_feed_eff"] = df["norm_feed_count"] / df["norm_sum_distance"]
    df["norm_step_to_feed_eff"] = df["norm_feed_count"] / df["norm_steps"]

    return df


# ==============================================================
# PLOTS
# ==============================================================

def plot_caption():
    plt.figtext(
        0.5, -0.04,
        "Each color corresponds to a distinct pathfinding algorithm.",
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
    fig, ax = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    for mode, grp in df.groupby("__mode__"):
        avg = grp.groupby(["pathfinding_mode","larva_id"]).agg(
            mean_hunger=("hunger_level", "mean"),
            min_hunger=("hunger_level", "min"),
            max_hunger=("hunger_level", "max"),
            dist=("distance_to_nest", "mean")
        )
        
        color = get_color(mode)
        ax[0].scatter(avg["dist"], avg["min_hunger"], label=f"{mode} (min)", color=color, marker="x")
        ax[1].scatter(avg["dist"], avg["mean_hunger"], label=f"{mode} (mean)", color=color)
        ax[2].scatter(avg["dist"], avg["max_hunger"], label=f"{mode} (max)", color=color, marker="s")

    ax[0].set_title("Larvae hunger vs distance to nest (min)")
    ax[1].set_title("Larvae hunger vs distance to nest (mean)")
    ax[2].set_title("Larvae hunger vs distance to nest (max)")

    for a in ax:
        a.set_xlabel("Distance to nest")
        a.set_ylabel("Average larvae hunger")
        a.legend()

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

def plot_foraging_efficiency():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    if df.empty:
        print("[SKIP] No aggregate data.")
        return

    if not {"mean_distance_per_feed_larvae", "mean_distance_per_feed_wasp", "num_wasps"}.issubset(df.columns):
        print("[WARN] Missing columns for normalization.")
        return

    # Added normalization: divide distance per feed by number of wasps
    df["mean_distance_per_feed_wasp"] = df["mean_distance_per_feed_wasp"] / df["num_wasps"].replace(0, np.nan)

    agg = df.groupby("__mode__", as_index=False).agg(
        mean_dist_larvae=("mean_distance_per_feed_larvae", "mean"),
        std_dist_larvae=("mean_distance_per_feed_larvae", "std"),
        mean_dist_wasp=("mean_distance_per_feed_wasp", "mean"),
        std_dist_wasp=("mean_distance_per_feed_wasp", "std"),
        mean_eff=("feeding_efficiency", "mean"),
        std_eff=("feeding_efficiency", "std")
    )

    x = np.arange(len(agg["__mode__"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, agg["mean_dist_larvae"], width,
           yerr=agg["std_dist_larvae"], label="Larvae Feed", color="#56B4E9", alpha=0.9, capsize=4)
    ax.bar(x + width/2, agg["mean_dist_wasp"], width,
           yerr=agg["std_dist_wasp"], label="Wasp Feed (per wasp)", color="#E69F00", alpha=0.9, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["__mode__"].str.capitalize(), rotation=15)
    ax.set_ylabel("Mean distance per feeding (normalized per wasp)")
    ax.set_title("Distance per feeding (Larvae vs Wasp) across algorithms")
    ax.legend()
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7_foraging_efficiency_compare.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(agg["__mode__"], agg["mean_eff"], yerr=agg["std_eff"],
            color=[COLORS.get(m, "gray") for m in agg["__mode__"]], capsize=4)
    plt.ylabel("Feeding efficiency")
    plt.title("Feeding efficiency by algorithm")
    plt.xticks(rotation=15)
    plot_caption()
    plt.tight_layout()
    plt.savefig(OUT / "7b_feeding_efficiency_by_algorithm.png", dpi=150, bbox_inches="tight")
    plt.close()


# 7️ Req 8: Foraging frequency
def plot_foraging_frequency():
    df = load_csvs("output_logs/*/aggregate_results_*.csv")
    if df.empty:
        print("[SKIP] No aggregate data for frequency.")
        return

    # Check both columns exist
    if not {"mean_feed_freq_wasp", "mean_feed_freq_larvae"}.issubset(df.columns):
        print("[WARN] Missing feed frequency columns in aggregate data.")
        print("Expected mean_feed_freq_wasp / mean_feed_freq_larvae")
        return

    # Group by mode to get mean/std
    agg = df.groupby("__mode__", as_index=False).agg(
        mean_feed_wasp=("mean_feed_freq_wasp", "mean"),
        std_feed_wasp=("std_feed_freq_wasp", "std"),
        mean_feed_larvae=("mean_feed_freq_larvae", "mean"),
        std_feed_larvae=("std_feed_freq_larvae", "std"),
    )

    print(agg[["__mode__", "mean_feed_wasp", "std_feed_wasp"]])


    # Plot side-by-side bars for comparison
    x = np.arange(len(agg["__mode__"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, agg["mean_feed_wasp"], width, 
           yerr=agg["std_feed_wasp"], 
           label="Wasp Feed Frequency", 
           color="#E69F00", alpha=0.9, capsize=4)
    ax.bar(x + width/2, agg["mean_feed_larvae"], width, 
           yerr=agg["std_feed_larvae"], 
           label="Larvae Feed Frequency", 
           color="#56B4E9", alpha=0.9, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["__mode__"].str.capitalize(), rotation=20)
    ax.set_ylabel("Feed frequency (mean ± std)")
    ax.set_title("Foraging frequency comparison: Wasp vs Larvae")
    ax.legend()
    plot_caption()
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

    plt.xlabel("Distance per wasp per step")
    plt.ylabel("Feeding per larva")
    plt.title("7c. Normalized: Distance per Wasp vs Feeding per Larva")
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
                     xerr=row["std_x"], yerr=row["std_y"],
                     fmt='o', color=color, label=row["__mode__"].capitalize(),
                     capsize=4)
    plt.xlabel("Distance per wasp per step")
    plt.ylabel("Feeding per larva")
    plt.title("7c1. Averaged: Distance per Wasp vs Feeding per Larva")
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

    plt.xlabel("Distance per wasp per step")
    plt.ylabel("Normalized feeding efficiency (feeds per distance)")
    plt.title("7d. Normalized: Walk-to-Feed Efficiency per Wasp per Step")
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
                     xerr=row["std_x"], yerr=row["std_y"],
                     fmt='o', color=color, label=row["__mode__"].capitalize(),
                     capsize=4)
    plt.xlabel("Distance per wasp per step")
    plt.ylabel("Normalized feeding efficiency (feeds per distance)")
    plt.title("7d1. Averaged: Walk-to-Feed Efficiency per Wasp per Step")
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

    plt.xlabel("Steps per wasp (normalized)")
    plt.ylabel("Feeding per larva")
    plt.title("7e. Normalized: Steps per Wasp vs Feeding per Larva")
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
                     xerr=row["std_x"], yerr=row["std_y"],
                     fmt='o', color=color, label=row["__mode__"].capitalize(),
                     capsize=4)
    plt.xlabel("Steps per wasp (normalized)")
    plt.ylabel("Feeding per larva")
    plt.title("7e1. Averaged: Steps per Wasp vs Feeding per Larva")
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
