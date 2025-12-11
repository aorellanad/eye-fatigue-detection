import cv2
import mediapipe as mp
import time
import csv
import json
from datetime import datetime
from scipy.spatial import distance
import os
import uuid
import statistics

# -------------------------
# Config
# -------------------------
session_id = str(uuid.uuid4())[:8]
os.makedirs("dataset", exist_ok=True)
CSV_FILE = f"dataset/blinks_dataset_session_{session_id}.csv"
SESSION_MINUTES = 5
EAR_THRESHOLD = 0.20
CAMERA_INDEX = 0

# -------------------------
# Start MediaPipe Face Mesh
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------
# Create CSV and write header
# -------------------------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'session_id', 'participant_id', 'date', 'time_of_day', 'minute_index',
            'blink_count', 'blink_rate', 'raw_blink_timestamps',
            'total_blinks', 'avg_blinks', 'std_blinks',
            'age', 'glasses', 'hours_sleep', 'caffeine_last_6h',
            'task', 'lighting',
            'self_report_score', 'self_report_label',
            'auto_score', 'auto_label',
            'notes'
        ])


# -------------------------
# Calculate EAR
# -------------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0


left_eye_idx = [33, 160, 158, 133, 153, 144]
right_eye_idx = [263, 387, 385, 362, 380, 373]


# -------------------------
# Draw eye contours on window
# -------------------------
def draw_eye_contours(frame, eye_points, color=(0, 255, 0)):
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 2, color, -1)
    if len(eye_points) > 1:
        pts = eye_points + [eye_points[0]]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 1)


# -------------------------
# Subjective label
# -------------------------
def map_self_label(score:int):
    if score <= 2: return "normal"
    if score == 3: return "moderated"
    return "tired"


# -------------------------
# Calculate auto score
# -------------------------
def score_from_avg(avg_blinks):
    if avg_blinks < 10:
        return 1
    elif avg_blinks < 20:
        return 2
    elif avg_blinks < 30:
        return 3
    elif avg_blinks < 40:
        return 4
    else:
        return 5


def penalty_from_std(std_blinks):
    if std_blinks < 3:
        return 0
    elif std_blinks < 6:
        return 1
    else:
        return 2


def score_to_label(score):
    if score <= 2:
        return "normal"
    elif score == 3:
        return "moderated"
    else:
        return "tired"


# -------------------------
# User metadata input
# -------------------------
print("\n--- User metadata (fill the fields) ---")
participant_id = input("User id (e.g: P01): ").strip()
age = input("Age (optional): ").strip()
glasses = input("Uses glasses? (yes/no): ").strip().lower()
hours_sleep = input("Hours of sleep last night (optional): ").strip()
caffeine = input("Had caffeine in last 6h? (yes/no): ").strip().lower()
task = input("Task (work/entertainment/other): ").strip()
lighting = input("Lighting (neutral/dim/bright/backlight): ").strip()
notes = input("Notes (optional): ").strip()

print(f"\nStarting session {session_id} for participant {participant_id}. Baseline 1 minute... (press 'q' to abort)\n")

cap = cv2.VideoCapture(CAMERA_INDEX)
time.sleep(1.0)

# -------------------------
# session variables
# -------------------------
baseline_blinks = 0
blink_in_progress = False
raw_timestamps_by_minute = []
raw_timestamps = []
per_minute_counts = []

baseline_end = time.time() + 60

# -------------------------
# Baseline 1 minute
# -------------------------
baseline_window = "Capture - Baseline (1 min)"
cv2.namedWindow(baseline_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(baseline_window, 960, 540)

print("--- Baseline 1 minute ---")

while time.time() < baseline_end:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        le = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in left_eye_idx]
        re = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in right_eye_idx]

        ear = (eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0

        # Draw eyes
        draw_eye_contours(frame, le, (0, 255, 0))
        draw_eye_contours(frame, re, (0, 255, 0))

        # Show EAR and baseline blinks
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Baseline blinks: {baseline_blinks}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)

        # Blink counting logic
        if ear < EAR_THRESHOLD and not blink_in_progress:
            blink_in_progress = True
            baseline_blinks += 1
        elif ear >= EAR_THRESHOLD:
            blink_in_progress = False
    else:
        # When no face detected
        cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show window of baseline
    cv2.imshow(baseline_window, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Aborted by user during baseline.")
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        exit(0)

# Close baseline window and proceed
cv2.destroyWindow(baseline_window)
print("Baseline has completed. Starting main session...\n")

# -------------------------
# Main session
# -------------------------
session_start = time.time()
session_end = session_start + (SESSION_MINUTES * 60)
current_minute_end = time.time() + 60
minute = 1
minute_blinks = 0
blink_in_progress = False

session_window = "Live blink detection session"
cv2.namedWindow(session_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(session_window, 960, 540)

while time.time() < session_end:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        le = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in left_eye_idx]
        re = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in right_eye_idx]

        ear = (eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0

        # Draw eyes
        draw_eye_contours(frame, le, (0, 255, 0))
        draw_eye_contours(frame, re, (0, 255, 0))

        # Show EAR, mins and blinks
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Minute: {minute}/{SESSION_MINUTES}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {minute_blinks}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show blink when it happens
        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "BLINK", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        # Blink counting logic
        if ear < EAR_THRESHOLD and not blink_in_progress:
            blink_in_progress = True
            minute_blinks += 1
            raw_timestamps_by_minute.append(time.time())
            raw_timestamps.append(time.time())
        elif ear >= EAR_THRESHOLD:
            blink_in_progress = False
    else:
        cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Check if minute ended
    if time.time() >= current_minute_end:
        per_minute_counts.append(minute_blinks)
        blink_rate = minute_blinks
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id, participant_id, datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"), minute,
                minute_blinks, blink_rate, json.dumps(raw_timestamps_by_minute),
                "", "", "",
                age, glasses, hours_sleep, caffeine,
                task, lighting,
                "", "", "", "",
                notes
            ])
        print(f"{minute} min: {minute_blinks} saved blinks.")
        minute += 1
        minute_blinks = 0
        raw_timestamps_by_minute = []
        current_minute_end += 60

    # Show session window
    cv2.imshow(session_window, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Aborted by user during session.")
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        exit(0)

cv2.destroyWindow(session_window)

# -------------------------
# Final calculations
# -------------------------
total_blinks = sum(per_minute_counts)
avg_blinks = total_blinks / len(per_minute_counts) if len(per_minute_counts) > 0 else 0
std_blinks = statistics.pstdev(per_minute_counts) if len(per_minute_counts) > 1 else 0

print(f"\nTotal blinks: {total_blinks}")
print(f"Average by min: {avg_blinks:.2f}")
print(f"Standard deviation: {std_blinks:.2f}")

# -------------------------
# Self-report score
# -------------------------
while True:
    try:
        score = int(input("\nSelf-report Likert (1-5): "))
        if 1 <= score <= 5:
            break
    except:
        pass

self_label = map_self_label(score)

# -------------------------
# Automatic score
# -------------------------
auto_score = score_from_avg(avg_blinks) + penalty_from_std(std_blinks)
auto_score = min(auto_score, 5)
auto_label = score_to_label(auto_score)

print("\n--- Complementary results ---")
print(f"Auto score: {auto_score}")
print(f"Auto score label: {auto_label}")

# -------------------------
# Save session summary
# -------------------------
with open(CSV_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        session_id, participant_id, datetime.now().strftime("%Y-%m-%d"),
        datetime.now().strftime("%H:%M:%S"), "session_summary",
        "", "", json.dumps(raw_timestamps),
        total_blinks, f"{avg_blinks:.2f}", f"{std_blinks:.2f}",
        age, glasses, hours_sleep, caffeine,
        task, lighting,
        score, self_label,
        auto_score, auto_label,
        f"baseline_blinks={baseline_blinks}; notes={notes}"
    ])

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\nSession data saved to:", CSV_FILE)
print(f"\n--- Finished session ---")
