import cv2
import mediapipe as mp
import time
import csv
import json
from datetime import datetime
from scipy.spatial import distance
import os
import uuid

# -------------------------
# Config
# -------------------------
os.makedirs("dataset", exist_ok=True)
BLINKS_BY_MINUTE_FILE = f"dataset/blinks_by_minute.csv"
SESSION_METADATA_FILE = f"dataset/session_metadata.csv"
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
# Create CSVs and write headers
# -------------------------
blinks_by_minute_file_exists_and_filled = not os.path.exists(BLINKS_BY_MINUTE_FILE) or os.path.getsize(
    BLINKS_BY_MINUTE_FILE) == 0

if blinks_by_minute_file_exists_and_filled:
    with open(BLINKS_BY_MINUTE_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([
            'session_id', 'participant_id', 'date', 'time_of_day', 'minute_index',
            'blink_count', 'blink_rate', 'raw_blink_timestamps'
        ])

session_metadata_file_exists_and_filled = not os.path.exists(SESSION_METADATA_FILE) or os.path.getsize(
    SESSION_METADATA_FILE) == 0

if session_metadata_file_exists_and_filled:
    with open(SESSION_METADATA_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([
            'session_id', 'participant_id', 'date', 'time_of_day', 'baseline_blinks', 'age', 'glasses', 'hours_sleep',
            'caffeine_last_6h', 'task', 'lighting', 'self_report_initial_score', 'self_report_final_score',
        ])


# -------------------------
# Calculate EAR
# -------------------------
def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c) if c != 0 else 0


left_eye_idx = [33, 160, 158, 133, 153, 144]
right_eye_idx = [263, 387, 385, 362, 380, 373]


# -------------------------
# Draw eye contours on window
# -------------------------
def draw_eye_contours(result_frame, eye_points, color=(0, 255, 0)):
    for (x, y) in eye_points:
        cv2.circle(result_frame, (x, y), 2, color, -1)
    if len(eye_points) > 1:
        pts = eye_points + [eye_points[0]]
        for i in range(len(pts) - 1):
            cv2.line(result_frame, pts[i], pts[i + 1], color, 1)


# -------------------------
# User metadata input
# -------------------------
session_id = str(uuid.uuid4())[:8]
print("\n--- User metadata (fill the fields) ---")
participant_id = input("User id (e.g: P01): ").strip()
age = input("Age (optional): ").strip()
glasses = input("Uses glasses? (yes/no): ").strip().lower()
hours_sleep = input("Hours of sleep last night (optional): ").strip()
caffeine = input("Had caffeine in last 6h? (yes/no): ").strip().lower()
task = input("Task (work/entertainment/other): ").strip()
lighting = input("Lighting (neutral/dim/bright/backlight): ").strip()
initial_score = int(
    input("Initial self-report Likert (0 - No fatigue, 1 - Slight fatigue, 2 - Moderated fatigue, 3 - High Fatigue): "))

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
        with open(BLINKS_BY_MINUTE_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                session_id, participant_id, datetime.now().strftime("%d/%m/%y"),
                datetime.now().strftime("%H:%M:%S"), minute,
                minute_blinks, json.dumps(raw_timestamps_by_minute),
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
# Self-report score
# -------------------------
while True:
    try:
        final_score = int(input(
            "\nfinal self-report Likert (0 - No fatigue, 1 - Slight fatigue, 2 - Moderated fatigue, 3 - High Fatigue): "))
        if 1 <= final_score <= 5:
            break
    except:
        pass

# -------------------------
# Save session summary
# -------------------------
with open(SESSION_METADATA_FILE, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow([
        session_id, participant_id, datetime.now().strftime("%d/%m/%y"),
        datetime.now().strftime("%H:%M:%S"), baseline_blinks, age, glasses, hours_sleep,
        caffeine, task, lighting, initial_score, final_score,
    ])

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\nSession blinks data saved to:", BLINKS_BY_MINUTE_FILE)
print("\nSession metadata saved to:", SESSION_METADATA_FILE)
print(f"\n--- Finished session ---")
