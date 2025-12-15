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
        avg_blinks_per_min=("blink_count", "mean"),
        std_blinks_per_min=("blink_count", "std"),
        session_minutes=("minute_index", "count")
    )
    .reset_index()
)

session_stats["std_blinks_per_min"] = session_stats["std_blinks_per_min"].fillna(0)

# =========================
# merge with metadata
# =========================
df = session_stats.merge(metadata_df, on="session_id", how="inner")

# =========================
# add blink baseline ratio
# =========================
df["blink_baseline_ratio"] = (
        df["avg_blinks_per_min"] /
        df["baseline_blinks"].replace(0, np.nan)
).fillna(0)

# =========================
# calculate subjective scores and labels
# =========================
df["subjective_delta_score"] = (
        df["self_report_score_final"] -
        df["self_report_score_initial"]
)


def delta_label(delta):
    if delta <= 0:
        return "no_change_or_improved"
    elif delta == 1:
        return "slight_worsening"
    else:
        return "significant_worsening"


df["subjective_delta_label"] = df["subjective_delta_score"].apply(delta_label)


# =========================
# auto scoring based on avg_blinks_per_min
# =========================
def auto_score(avg_blinks):
    if avg_blinks < 10:
        return 1
    elif avg_blinks <= 20:
        return 2
    elif avg_blinks <= 30:
        return 3
    elif avg_blinks <= 40:
        return 4
    else:
        return 5


def auto_label(score):
    if score <= 2:
        return "Normal"
    elif score == 3:
        return "Moderate"
    else:
        return "Tired"


df["auto_score"] = df["avg_blinks_per_min"].apply(auto_score)
df["auto_label"] = df["auto_score"].apply(auto_label)

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
        "avg_blinks_per_min",
        "std_blinks_per_min",
        "blink_baseline_ratio",

        "self_report_score_initial",
        "self_report_label_initial",
        "self_report_score_final",
        "self_report_label_final",

        "subjective_delta_score",
        "subjective_delta_label",

        "auto_score",
        "auto_label"
    ]
]

# =========================
# save training dataset
# =========================
training_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Training dataset generated")
print("Saved at:", OUTPUT_FILE)
training_df.head()
