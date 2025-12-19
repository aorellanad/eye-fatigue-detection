import pandas as pd
import numpy as np

# =========================
# config
# =========================
DATASET_PATH = "dataset"
BLINKS_FILE = f"{DATASET_PATH}/blinks_by_minute.csv"
METADATA_FILE = f"{DATASET_PATH}/session_metadata.csv"
OUTPUT_FILE = f"{DATASET_PATH}/training_dataset.csv"

# =========================
# load data
# =========================
blinks_df = pd.read_csv(BLINKS_FILE, sep=";")
metadata_df = pd.read_csv(METADATA_FILE, sep=";")

# =========================
# calculate session statistics
# =========================
session_stats = (
    blinks_df
    .groupby("session_id")
    .agg(
        total_blinks=("blink_count", "sum"),
        avg_blinks=("blink_count", "mean"),
        std_blinks=("blink_count", "std"),
        session_minutes=("minute_index", "count")
    )
    .reset_index()
)


# =========================
# fill 0 for std_blinks NaN values in case of sessions with only 1 minute
# =========================
session_stats["std_blinks"] = session_stats["std_blinks"].fillna(0)


# =========================
# round avg_blinks and std_blinks
# =========================
session_stats["avg_blinks"] = session_stats["avg_blinks"].round(2)
session_stats["std_blinks"] = session_stats["std_blinks"].round(2)

# =========================
# merge with metadata
# =========================
df = session_stats.merge(metadata_df, on="session_id", how="inner")

# =========================
# add blink baseline ratio
# =========================
df["blink_baseline_ratio"] = (
        df["avg_blinks"] /
        df["baseline_blinks"].replace(0, np.nan)
).fillna(0)

df["blink_baseline_ratio"] = df["blink_baseline_ratio"].round(2)

# =========================
# calculate subjective scores and labels
# =========================
df["subjective_delta_score"] = (
        df["self_report_final_score"] -
        df["self_report_initial_score"]
)


# =========================
# auto scoring based on avg_blinks
# =========================
def auto_score(avg_blinks):
    if avg_blinks < 10:
        return 0
    elif avg_blinks <= 20:
        return 1
    elif avg_blinks <= 30:
        return 2
    else:
        return 3

df["auto_score"] = df["avg_blinks"].apply(auto_score)

# =========================
# build training dataset
# =========================
training_df = df[
    [
        "session_id",
        "participant_id",

        "age",
        "glasses",
        "hours_sleep",
        "caffeine_last_6h",
        "task",
        "lighting",

        "baseline_blinks",
        "session_minutes",
        "total_blinks",
        "avg_blinks",
        "std_blinks",
        "blink_baseline_ratio",

        "self_report_initial_score",
        "self_report_final_score",
        "subjective_delta_score",
        "auto_score",
    ]
]

# =========================
# save training dataset
# =========================
training_df.to_csv(OUTPUT_FILE, index=False, sep=";")

print("✅ Training dataset generated")
print("Saved at:", OUTPUT_FILE)
training_df.head()
