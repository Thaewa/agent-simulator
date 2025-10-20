# analyze_logs.py
# Analyze simulation CSV logs (agent-level and nest-level)
# Thaewa Tansarn — Capstone Data Mining & Visualization Module

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Step 0: Ensure output directory exists
OUTPUT_DIR = "../analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Locate the latest simulation logs
agent_files = sorted(glob.glob(os.path.join("..", "agent_log_*.csv")))
nest_files = sorted(glob.glob(os.path.join("..", "nest_log_*.csv")))

if not agent_files or not nest_files:
    raise FileNotFoundError("No simulation log files found in parent directory.")

agent_path = agent_files[-1]
nest_path = nest_files[-1]

print(f"Using logs:\n  Agent: {agent_path}\n  Nest:  {nest_path}")

# Step 2: Load data
df_agent = pd.read_csv(agent_path)
df_nest = pd.read_csv(nest_path)

print("\n Loaded successfully!")
print(f"Agent log shape: {df_agent.shape}")
print(f"Nest log shape:  {df_nest.shape}")

# Step 3: Quick overview
print("\nAgent log columns:", df_agent.columns.tolist())
print("Nest log columns:", df_nest.columns.tolist())

# Step 4: Plot hunger & food balance over time
plt.figure(figsize=(10, 6))
plt.plot(df_nest['timestamp'], df_nest['avg_hunger_foragers'], label='Foragers Hunger', color='tab:red')
plt.plot(df_nest['timestamp'], df_nest['avg_hunger_feeders'], label='Feeders Hunger', color='tab:orange')
plt.xlabel("Timestep")
plt.ylabel("Average Hunger Level")
plt.title("Hunger Dynamics Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hunger_dynamics.png"), dpi=300)
plt.show()

# Step 5: Correlation matrix (optional)
numeric_cols = df_nest.select_dtypes('number')
corr = numeric_cols.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Nest-level)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.show()

# Step 6: Save summary statistics
summary = df_nest.describe()
summary_path = os.path.join(OUTPUT_DIR, "nest_summary_stats.csv")
summary.to_csv(summary_path, index=True)
print(f"\n Summary statistics exported to {summary_path}")

# Step 7: Moving average smoothing (window=20)
df_nest["hunger_foragers_smooth"] = df_nest["avg_hunger_foragers"].rolling(window=20).mean()
df_nest["hunger_feeders_smooth"] = df_nest["avg_hunger_feeders"].rolling(window=20).mean()

plt.figure(figsize=(10, 6))
plt.plot(df_nest["timestamp"], df_nest["hunger_foragers_smooth"], label="Foragers (Smoothed)", color="tab:red")
plt.plot(df_nest["timestamp"], df_nest["hunger_feeders_smooth"], label="Feeders (Smoothed)", color="tab:orange")
plt.xlabel("Timestep")
plt.ylabel("Average Hunger (Smoothed)")
plt.title("Hunger Trend (Moving Average = 20)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "moving_average.png"), dpi=300)
plt.show()

# Step 8: Histogram distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_nest["avg_hunger_foragers"], color="tab:red", bins=30, label="Foragers", kde=True)
sns.histplot(df_nest["avg_hunger_feeders"], color="tab:orange", bins=30, label="Feeders", kde=True)
plt.xlabel("Average Hunger")
plt.title("Hunger Distribution Across Simulation")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hunger_distribution.png"), dpi=300)
plt.show()

# Step 9: Role Dynamics Over Time
# ---------------------------------
print("\n Analyzing role distribution over time...")

# Detect column used as 'role' substitute
role_col = None
for candidate in ["agent_role", "role"]:
    if candidate in df_agent.columns:
        role_col = candidate
        break

if role_col is None:
    raise KeyError("❌ No role column found in agent log. Expected one of: 'agent_role' or 'role'.")

print(f"Using column '{role_col}' for role grouping.")

# Count number of agents per role at each timestep
role_counts = df_agent.groupby(["timestamp", role_col]).size().unstack(fill_value=0)

if role_counts.empty:
    print(" Role count DataFrame is empty — no roles detected.")
else:
    plt.figure(figsize=(10,6))
    for col in role_counts.columns:
        plt.plot(role_counts.index, role_counts[col], label=str(col))
    plt.xlabel("Timestep")
    plt.ylabel("Agent Count")
    plt.title("Role Counts Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "role_counts_over_time.png")
    plt.savefig(output_path, dpi=300)
    print(f" Saved role dynamics plot → {output_path}")
    plt.show()



# Step 10: PCA Visualization for Behavioral Correlation
# ------------------------------------------------------
print("\n Running PCA for multi-variable behavioral correlation...")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

numeric_cols = df_nest.select_dtypes('number').dropna(axis=1)
X = StandardScaler().fit_transform(numeric_cols)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sc = plt.scatter(
    pca_result[:,0], pca_result[:,1],
    c=df_nest["avg_hunger_foragers"], cmap="coolwarm", s=20, alpha=0.7
)
plt.colorbar(sc, label="Avg Hunger (Foragers)")
plt.title("PCA Projection of Nest-level Behavioral Variables")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_behavioral_map.png"), dpi=300)
plt.show()
plt.close()

print(" PCA plot saved.")


# Step 11: Cross-correlation (Hunger → Feeding)
# -----------------------------------------------
print("\n Evaluating cross-correlation between hunger and feeding events...")

from scipy.signal import correlate

hunger_series = df_nest["avg_hunger_foragers"] - df_nest["avg_hunger_foragers"].mean()
feeding_series = df_nest["feeding_events"] - df_nest["feeding_events"].mean()

corr = correlate(hunger_series, feeding_series, mode='full')
lags = range(-len(df_nest)+1, len(df_nest))

plt.figure(figsize=(10,6))
plt.plot(lags, corr, color="tab:purple")
plt.title("Cross-correlation: Forager Hunger vs Feeding Events")
plt.xlabel("Lag (timesteps)")
plt.ylabel("Correlation")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_correlation_hunger_vs_feeding.png"), dpi=300)
plt.show()
plt.close()
print(" Cross-correlation analysis completed and saved.")


print(f"\n All figures saved in '{OUTPUT_DIR}/' successfully.")
