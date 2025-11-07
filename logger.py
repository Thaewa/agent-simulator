import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd
import glob

class DataLogger:
    """
    Handles all logs: agent, nest, larvae, and aggregate.
    Each pathfinding mode gets its own folder under output_logs/.
    Agent, nest, and larvae logs are reset each run, 
    while the aggregate log persists and keeps appending results.
    """

    def __init__(self, pathfinding_mode="unknown", simulation_id=None, reset_logs=True):
        self.pathfinding_mode = pathfinding_mode.lower()

        # --- paths ---
        root_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(root_dir, "output_logs", self.pathfinding_mode)
        os.makedirs(base_dir, exist_ok=True)

        self.agent_log_path     = os.path.join(base_dir, f"agent_log_{self.pathfinding_mode}.csv")
        self.nest_log_path      = os.path.join(base_dir, f"nest_log_{self.pathfinding_mode}.csv")
        self.larvae_log_path    = os.path.join(base_dir, f"larvae_log_{self.pathfinding_mode}.csv")
        self.aggregate_log_path = os.path.join(base_dir, f"aggregate_results_{self.pathfinding_mode}.csv")

        # --- headers ---
        self.agent_headers = ["simulation_id","timestamp","agent_id","agent_role","action","target_id",
                              "position_x","position_y","hunger_level","hunger_cue","food_stored",
                              "nest_layer","larvae_hunger_avg","total_food_in_nest",
                              "rush_intensity","exploration_bias","pathfinding_mode"]
        self.nest_headers = ["simulation_id","timestamp","total_foragers","total_feeders","foraging_events",
                             "feeding_events","transfer_events","avg_hunger_foragers","avg_hunger_feeders",
                             "avg_hunger_larvae","food_balance_in_nest","rush_intensity","exploration_bias",
                             "active_cells","nest_size","pathfinding_mode"]
        self.larvae_headers = ["simulation_id","timestamp","larva_id","position_x","position_y",
                               "hunger_level","food_received","distance_to_nest","pathfinding_mode"]

        # --- simulation_id policy ---
        if simulation_id is not None:
            # ใช้ค่าที่ batch ส่งมาแบบตรงๆ ไม่ขยับ
            self.run_counter = int(simulation_id)
        else:
            # ไม่มี sim-id → auto-continue จาก aggregate ล่าสุด + 1
            last_id = 0
            if os.path.exists(self.aggregate_log_path):
                with open(self.aggregate_log_path, "r", newline="", encoding="utf-8") as f:
                    r = csv.reader(f); next(r, None)
                    rows = list(r)
                    if rows:
                        try:
                            last_id = int(rows[-1][0])
                        except:
                            last_id = 0
            self.run_counter = last_id + 1

        print(f"[INFO] New simulation started: {self.pathfinding_mode} (simulation_id={self.run_counter})")

        # --- prepare log files (reset or append) ---
        def ensure(path, headers):
            if reset_logs and os.path.exists(path):
                os.remove(path)
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(headers)

        ensure(self.agent_log_path,  self.agent_headers)
        ensure(self.nest_log_path,   self.nest_headers)
        ensure(self.larvae_log_path, self.larvae_headers)
    # -------------------------------
    # Logging functions
    # -------------------------------

    def log_agent(self, **kwargs):
        """Append a single agent event to the agent log."""
        kwargs["pathfinding_mode"] = self.pathfinding_mode
        kwargs["simulation_id"] = self.run_counter  # ✅ Add this
        with open(self.agent_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.agent_headers)
            writer.writerow(kwargs)

    def log_nest(self, **kwargs):
        """Append a single nest event to the nest log."""
        kwargs["pathfinding_mode"] = self.pathfinding_mode
        kwargs["simulation_id"] = self.run_counter  # ✅ Add this
        with open(self.nest_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.nest_headers)
            writer.writerow(kwargs)

    def log_larvae(self, **kwargs):
        """Append a single larvae event to the larvae log."""
        kwargs["pathfinding_mode"] = self.pathfinding_mode
        kwargs["simulation_id"] = self.run_counter  # ✅ Add this
        with open(self.larvae_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.larvae_headers)
            writer.writerow(kwargs)

    def log_aggregate(self, results, config):
        """
        Append one aggregate result to the aggregate log.

        - Automatically adapts header if new columns appear.
        - Never overwrites existing results.
        - Converts numpy types to standard Python types.
        - Appends timestamp for every run.
        """

        clean_results = {}
        for k, v in results.items():
            if isinstance(v, (np.generic, float, int)):
                clean_results[k] = round(float(v), 6)
            else:
                clean_results[k] = v

        # increment simulation run counter
        #self.run_counter += 1
        clean_results["simulation_id"] = self.run_counter
        clean_results["pathfinding_mode"] = self.pathfinding_mode
        clean_results["timestamp_recorded"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        file_path = self.aggregate_log_path

        existing_rows = []
        existing_header = []

        # read existing file if exists
        if os.path.exists(file_path):
            with open(file_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_header = reader.fieldnames or []
                existing_rows = list(reader)

        # merge headers to allow new columns
        all_fields = list(dict.fromkeys(existing_header + list(clean_results.keys())))

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()

            for row in existing_rows:
                writer.writerow(row)

            writer.writerow(clean_results)

        print(f"[INFO] Aggregate log updated → {os.path.basename(file_path)} (run {self.run_counter})")

    def debug_view_aggregate(self, mode=None, n=10):
        """
        Inspect a single aggregate_results_<mode>.csv:
        - show filepath found
        - show first rows and dtypes
        - show per-column numeric-conversion preview
        - print simple means of key columns
        """
        import pandas as pd, numpy as np, glob, os

        root_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(root_dir, "output_logs")
        pattern = os.path.join(output_dir, "*/aggregate_results_*.csv")
        files = sorted(glob.glob(pattern))

        print(f"[DEBUG] Found {len(files)} aggregate files.")
        if not files:
            return

        # pick by mode if given, otherwise first file
        pick = None
        if mode:
            for fp in files:
                folder = os.path.basename(os.path.dirname(fp))
                if folder.lower() == mode.lower():
                    pick = fp
                    break
            if pick is None:
                print(f"[DEBUG] No file found for mode='{mode}'.")
                return
        else:
            pick = files[0]

        print(f"[DEBUG] Inspect file: {pick}")
        df = pd.read_csv(pick, encoding="utf-8-sig", engine="python")
        print(f"[DEBUG] shape={df.shape}")
        print("[DEBUG] head:")
        print(df.head(n))

        print("\n[DEBUG] dtypes:")
        print(df.dtypes)

        # try numeric conversion on all columns except obvious non-numeric
        nn = ["pathfinding_mode", "timestamp_recorded"]
        for c in df.columns:
            if c not in nn:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # show quick stats for key columns if exist
        keys = ["mean_hunger_larvae","feeding_efficiency",
                "sum_distance_traveled_feeders","avg_distance_traveled_feeders"]
        print("\n[DEBUG] means (after numeric coerce):")
        for k in keys:
            if k in df.columns:
                print(f"  {k}: mean={df[k].mean()}  count={df[k].count()}")

        # also show unique pathfinding_mode
        if "pathfinding_mode" in df.columns:
            print("\n[DEBUG] unique modes in file:", df["pathfinding_mode"].unique())


