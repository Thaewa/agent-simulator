"""Versioned run metrics with explicit denominators and activity-time roles."""

import numpy as np


METRICS_SCHEMA = "activity_roles_v2"


def ratio(numerator, denominator):
    """Undefined denominators are missing values, not perfect zero costs."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def distance_per_event(wasps, counter):
    return ratio(sum(w.distance_traveled for w in wasps),
                 sum(getattr(w, counter) for w in wasps))


def summarize_agents(wasps, larvae, steps):
    result = {"metrics_schema": METRICS_SCHEMA, "simulation_steps": steps,
              "total_wasp_distance": sum(w.distance_traveled for w in wasps),
              "total_wasp_steps": sum(w.step_count for w in wasps)}
    for label, counter in (("wasp", "feed_count_wasp"), ("larvae", "feed_count_larvae")):
        counts = [getattr(w, counter) for w in wasps]
        mean = float(np.mean(counts)) if counts else float("nan")
        std = float(np.std(counts)) if counts else float("nan")
        result[f"mean_feed_count_{label}"] = mean
        result[f"std_feed_count_{label}"] = std
        result[f"mean_feed_rate_{label}"] = ratio(mean, steps)
        result[f"std_feed_rate_{label}"] = ratio(std, steps)
        # Compatibility names now represent actual rates, identified by schema.
        result[f"mean_feed_freq_{label}"] = result[f"mean_feed_rate_{label}"]
        result[f"std_feed_freq_{label}"] = result[f"std_feed_rate_{label}"]
    participants = [w for w in wasps if w.feeder_step_count > 0 or w.feeder_feed_count_larvae > 0]
    distance = sum(w.feeder_distance_traveled for w in wasps)
    events = sum(w.feeder_feed_count_larvae for w in wasps)
    exposure = sum(w.feeder_step_count for w in wasps)
    result.update({
        "num_feeder_participants": len(participants),
        "sum_distance_traveled_feeders": distance,
        "sum_feed_count_larvae": events,
        "sum_step_count_feeders": exposure,
        "avg_distance_traveled_feeders": ratio(distance, len(participants)),
        "avg_feed_count_larvae": ratio(events, len(participants)),
        "avg_step_count_feeders": ratio(exposure, len(participants)),
        "efficiency_walk_feed_ratio": ratio(events, distance),
        "efficiency_step_feed_ratio": ratio(events, exposure),
        "mean_distance_per_feed_larvae": distance_per_event(wasps, "feed_count_larvae"),
        "mean_distance_per_feed_wasp": distance_per_event(wasps, "feed_count_wasp"),
        "final_low_hunger_fraction": ratio(sum(a.hunger < 1.5 for a in larvae), len(larvae)),
        "low_hunger_threshold": 1.5,
    })
    result["feeding_efficiency"] = result["final_low_hunger_fraction"]
    return result


def validate_schema(df):
    if "metrics_schema" not in df or not df.metrics_schema.eq(METRICS_SCHEMA).all():
        raise ValueError("Historical metrics require their original plotting script. "
                         "Run python run_experiment.py to produce activity_roles_v2 metrics.")
    if df.duplicated(["__mode__", "simulation_id"]).any():
        raise ValueError("Duplicate mode/run IDs: do not silently average separate runs")


def normalize_aggregate(df):
    if df.empty:
        return df
    validate_schema(df)
    df = df.copy()
    participants = df.num_feeder_participants.replace(0, np.nan)
    distance = df.sum_distance_traveled_feeders
    exposure = df.sum_step_count_feeders
    events = df.sum_feed_count_larvae
    df["norm_sum_distance"] = distance / participants
    df["norm_feed_count"] = events / df.num_larvae.replace(0, np.nan)
    df["norm_steps"] = exposure / participants
    df["distance_per_wasp_per_step"] = distance / exposure.replace(0, np.nan)
    df["feed_per_larva"] = df["norm_feed_count"]
    df["norm_walk_to_feed_eff"] = events / distance.replace(0, np.nan)
    df["norm_step_to_feed_eff"] = events / exposure.replace(0, np.nan)
    return df


def summarize_runs(df, columns):
    """Each row is one run. Error bars are SD of run-level metric values."""
    validate_schema(df)
    grouped = df.groupby("__mode__")[columns].agg(["mean", "std", "count"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()
