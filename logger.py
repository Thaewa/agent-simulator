import csv
import os
from datetime import datetime
import numpy as np
import pandas as pd
import glob


def validate_food_event_log(logger):
    """Reconcile the new journal with independently logged per-step event counts."""
    events = pd.read_csv(logger.event_log_path)
    nest = pd.read_csv(logger.nest_log_path)
    if events.event_id.tolist() != list(range(1, len(events) + 1)):
        raise ValueError("Food event IDs are not sequential for this run")
    if not events.empty:
        if not events.simulation_id.eq(logger.run_counter).all() or not events.pathfinding_mode.eq(logger.pathfinding_mode).all():
            raise ValueError("Food event run identity does not match logger")
        if not events.event_schema.eq("food_events_v1").all() or not events.amount_unit.eq("model_food").all():
            raise ValueError("Unexpected food event schema or units")
        if not (np.isfinite(events.amount).all() and events.amount.gt(0).all()):
            raise ValueError("Invalid successful food quantity")
        if not events.timestamp.isin(nest.timestamp[nest.timestamp.gt(0)]).all():
            raise ValueError("Food event outside completed simulation steps")
        if not events.timestamp.is_monotonic_increasing:
            raise ValueError("Food events are not in time order")
        for _, group in events.groupby("timestamp", sort=False):
            if group.step_event_order.tolist() != list(range(1, len(group) + 1)):
                raise ValueError("Food event within-step order is not sequential")
    kinds = {"larval_feed": "feeding_events", "adult_transfer": "transfer_events",
             "collection": "foraging_events"}
    if not events.event_type.isin(kinds).all():
        raise ValueError("Unexpected food event type")
    for kind, column in kinds.items():
        counts = events[events.event_type == kind].groupby("timestamp").size()
        actual = nest.timestamp.map(counts).fillna(0).to_numpy()
        if not np.array_equal(actual, nest[column].to_numpy()):
            raise ValueError(f"Food journal disagrees with nest log: {kind}")


class DataLogger:
    """
    Handles all logs: agent, nest, larvae, and aggregate.
    Each pathfinding mode gets its own folder under output_logs/.
    Agent, nest, and larvae logs are reset each run, 
    while the aggregate log persists and keeps appending results.
    """

    def __init__(self, pathfinding_mode="unknown", simulation_id=None, reset_logs=True,sensitivity=False, *, output_dir=None, enable_event_log=True):
        self.pathfinding_mode = pathfinding_mode.lower()
        # --- paths ---
        root_dir = os.path.dirname(os.path.abspath(__file__))
        logs_root = output_dir if output_dir is not None else os.path.join(
            root_dir, "output_logs" if not sensitivity else "output_logs_sensitivity"
        )
        base_dir = os.path.join(logs_root, self.pathfinding_mode)
        os.makedirs(base_dir, exist_ok=True)

        self.agent_log_path     = os.path.join(base_dir, f"agent_log_{self.pathfinding_mode}.csv")
        self.nest_log_path      = os.path.join(base_dir, f"nest_log_{self.pathfinding_mode}.csv")
        self.larvae_log_path    = os.path.join(base_dir, f"larvae_log_{self.pathfinding_mode}.csv")
        self.aggregate_log_path = os.path.join(base_dir, f"aggregate_results_{self.pathfinding_mode}.csv")
        self.event_log_path = os.path.join(base_dir, f"food_event_log_{self.pathfinding_mode}.csv")
        self.enable_event_log = enable_event_log
        self._event_id = 0
        self._event_timestamp = None
        self._step_event_order = 0
        self.event_headers = [
            "event_schema", "simulation_id", "pathfinding_mode", "event_id",
            "timestamp", "step_event_order", "event_type", "actor_id", "actor_type",
            "actor_role", "actor_x", "actor_y", "recipient_id", "recipient_type",
            "recipient_role", "recipient_x", "recipient_y", "amount", "amount_unit",
            "source_x", "source_y",
        ]

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
        if self.enable_event_log:
            ensure(self.event_log_path, self.event_headers)
    # -------------------------------
    # Logging functions
    # -------------------------------

    def log_food_event(self, timestamp, event_type, actor, recipient, amount, source_position=None):
        """Write one successful action immediately, before later movement/role changes."""
        if not self.enable_event_log:
            return
        if event_type not in {"collection", "adult_transfer", "larval_feed"}:
            raise ValueError("Unknown food event type")
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("Successful food events require a finite positive amount")
        self._event_id += 1
        if timestamp != self._event_timestamp:
            self._event_timestamp = timestamp
            self._step_event_order = 0
        self._step_event_order += 1
        row = dict.fromkeys(self.event_headers, "")
        row.update(event_schema="food_events_v1", simulation_id=self.run_counter,
                   pathfinding_mode=self.pathfinding_mode, event_id=self._event_id,
                   timestamp=timestamp, step_event_order=self._step_event_order,
                   event_type=event_type, amount=amount, amount_unit="model_food")
        for prefix, entity in (("actor", actor), ("recipient", recipient)):
            if entity is not None:
                row.update({f"{prefix}_id": entity.id, f"{prefix}_type": entity.type.value,
                            f"{prefix}_role": getattr(getattr(entity, "role", None), "name", ""),
                            f"{prefix}_x": entity.x, f"{prefix}_y": entity.y})
        if source_position is not None:
            row.update(source_x=source_position[0], source_y=source_position[1])
        with open(self.event_log_path, "a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.event_headers).writerow(row)


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


