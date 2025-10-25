# logger.py
import csv
import os
from datetime import datetime

class DataLogger:
    """
    Handles event-level (agent) and nest-level logging for the simulation.
    Automatically creates CSV files with headers.
    """

    def __init__(self, agent_log_path='agent_log.csv', nest_log_path='nest_log.csv'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(agent_log_path)[0]
        self.agent_log_path = f"{base}_{timestamp}.csv"
        base = os.path.splitext(nest_log_path)[0]
        self.nest_log_path = f"{base}_{timestamp}.csv"

        self.agent_headers = [
            "timestamp","agent_id","agent_role","action","target_id",
            "position_x","position_y","hunger_level","hunger_cue","food_stored",
            "nest_layer","larvae_hunger_avg","total_food_in_nest",
            "rush_intensity","exploration_bias"
        ]
        self.nest_headers = [
            "timestamp","total_foragers","total_feeders","foraging_events",
            "feeding_events","transfer_events","avg_hunger_foragers",
            "avg_hunger_feeders","avg_hunger_larvae","food_balance_in_nest",
            "rush_intensity","exploration_bias","active_cells","nest_size"
        ]

        # Initialize CSVs
        with open(self.agent_log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.agent_headers)
        with open(self.nest_log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.nest_headers)

    def log_agent(self, **kwargs):
        """Record one agent-level event."""
        with open(self.agent_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.agent_headers)
            writer.writerow(kwargs)

    def log_nest(self, **kwargs):
        """Record one nest-level summary snapshot."""
        with open(self.nest_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.nest_headers)
            writer.writerow(kwargs)

    # ======================================================================
    # Added for larvae logging & aggregate metrics
    # ----------------------------------------------------------------------
    # Adds new logs: larvae_log_*.csv and aggregate_results_*.csv
    # These enable per-larva hunger tracking and per-simulation summary stats.
    # ======================================================================

    def _init_larvae_logs(self):
        """Initialize additional logs for larvae and aggregate summaries."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "output_logs"
        os.makedirs(base_dir, exist_ok=True)

        self.larvae_log_path = os.path.join(base_dir, f"larvae_log_{timestamp}.csv")
        self.aggregate_log_path = os.path.join(base_dir, f"aggregate_results_{timestamp}.csv")

        self.larvae_headers = [
            "timestamp","larva_id","position_x","position_y",
            "hunger_level","food_received","distance_to_nest"
        ]
        self.aggregate_headers = [
            "simulation_id","pathfinding_mode",
            "mean_hunger_larvae","max_hunger_larvae","min_hunger_larvae",
            "mean_distance_per_feed","feeding_efficiency",
             "mean_feed_freq", "std_feed_freq",
             "num_wasps","num_larvae","wasp_to_larvae_ratio" # New columns
        ]

        # Create CSV files
        with open(self.larvae_log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.larvae_headers)
        with open(self.aggregate_log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.aggregate_headers)

    def log_larvae(self, **kwargs):
        """Record per-timestep larvae states."""
        if not hasattr(self, "larvae_log_path"):
            self._init_larvae_logs()
        with open(self.larvae_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.larvae_headers)
            writer.writerow(kwargs)

    def log_aggregate(self, **kwargs):
        """Record per-simulation summary statistics."""
        if not hasattr(self, "aggregate_log_path"):
            self._init_larvae_logs()

        with open(self.aggregate_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.aggregate_headers)
            writer.writerow(kwargs)
