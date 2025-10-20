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
            "position_x","position_y","hunger_level","food_stored",
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
