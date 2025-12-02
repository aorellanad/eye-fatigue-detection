import cv2
import mediapipe as mp
import time
import csv
import json
from datetime import datetime
from scipy.spatial import distance
import os
import uuid

# ----- Config -----
CSV_FILE = "dataset/blinks_dataset_stage2.csv"
SESSION_MINUTES = 5  # cambiar si deseas 10
EAR_THRESHOLD = 0.20  # ajustar en piloto si es necesario
CAMERA_INDEX = 0

# ----- Inicializar Mediapipe -----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ----- Asegurar header CSV -----
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id','participant_id','date','time_of_day','minute_index',
                         'blink_count','blink_rate','raw_blink_timestamps','total_blinks',
                         'avg_blinks','std_blinks','age','glasses','hours_sleep','caffeine_last_6h',
                         'task','lighting','self_report_score','self_report_label','notes'])

# ----- Funciones -----
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

def map_self_label(score):
    if score <= 2: return "normal"
    if score == 3: return "moderado"
    return "cansado"

# ----- Inputs del participante -----
participant_id = input("Participant ID (ej: P01): ").strip()
age = input("Age (optional): ").strip()
glasses = input("Uses glasses? (yes/no): ").strip().lower()
hours_sleep = input("Hours of sleep last night (optional): ").strip()
caffeine = input("Had caffeine in last 6h? (yes/no): ").strip().lower()
task = input("Task (reading/programming/other): ").strip()
lighting = input("Lighting (neutral/dim/bright/contraluz): ").strip()
notes = input("Notes (optional): ").strip()

session_id = str(uuid.uuid4())[:8]
print(f"\nStarting session {session_id} for participant {participant_id}. Baseline 1 minute...")

cap = cv2.VideoCapture(CAMERA_INDEX)
time.sleep(1.0)

# Baseline minute (optional)
baseline_blinks = 0
baseline_end = time.time() + 60
blink_in_progress = False
left_eye_idx = [33, 160, 158, 133, 153, 144]
right_eye_idx = [263, 387, 385, 362, 380, 373]
raw_timestamps = []  # all blinks in session
per_minute_counts = []

# Run baseline minute (optional, counts but not saved as session minute)
while time.time() < baseline_end:
    ret, frame = cap.read()
    if not ret:
        continue
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0]
        le = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in left_eye_idx]
        re = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in right_eye_idx]
        ear = (eye_aspect_ratio(le) + eye_aspect_ratio(re))/2.0
        if ear < EAR_THRESHOLD and not blink_in_progress:
            blink_in_progress = True
            baseline_blinks += 1
        elif ear >= EAR_THRESHOLD:
            blink_in_progress = False

print("Baseline done. Proceeding to main session...")

# Main session (per-minute capture)
session_start = time.time()
minute = 1
session_end = session_start + (SESSION_MINUTES * 60)
blink_in_progress = False
current_minute_end = time.time() + 60
minute_blinks = 0

while time.time() < session_end:
    ret, frame = cap.read()
    if not ret:
        continue
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0]
        le = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in left_eye_idx]
        re = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in right_eye_idx]
        ear = (eye_aspect_ratio(le) + eye_aspect_ratio(re))/2.0
        if ear < EAR_THRESHOLD and not blink_in_progress:
            blink_in_progress = True
            minute_blinks += 1
            raw_timestamps.append(time.time())
        elif ear >= EAR_THRESHOLD:
            blink_in_progress = False

    # When a minute ends, save row
    if time.time() >= current_minute_end:
        blink_rate = minute_blinks  # per minute
        per_minute_counts.append(minute_blinks)
        # write temporary row (total_blinks/avg will be filled at session end)
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, participant_id, datetime.now().strftime("%Y-%m-%d"),
                             datetime.now().strftime("%H:%M:%S"), minute,
                             minute_blinks, blink_rate, json.dumps(raw_timestamps),
                             "", "", "", age, glasses, hours_sleep, caffeine,
                             task, lighting, "", "", notes])
        print(f"Saved minute {minute}: blinks={minute_blinks}")
        # reset for next minute
        minute += 1
        minute_blinks = 0
        current_minute_end += 60

# End session: compute totals and ask Likert
total_blinks = sum(per_minute_counts)
avg_blinks = total_blinks / len(per_minute_counts)
import statistics
std_blinks = statistics.pstdev(per_minute_counts) if len(per_minute_counts)>1 else 0

print(f"\nSession finished. Total blinks: {total_blinks}, avg/min: {avg_blinks:.2f}")
# Likert survey
while True:
    try:
        score = int(input("Self-report Likert (1-5) — 1=Nothing, 5=Extreme: ").strip())
        if 1 <= score <= 5:
            break
    except:
        pass
print("Thank you. Saving final summary...")

# Update the last SESSION_MINUTES rows with totals/avg/std/self_report
# Simple approach: append a session summary row (instead of overwriting minute rows)
with open(CSV_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([session_id, participant_id, datetime.now().strftime("%Y-%m-%d"),
                     datetime.now().strftime("%H:%M:%S"), "session_summary",
                     "", "", json.dumps(raw_timestamps),
                     total_blinks, f"{avg_blinks:.2f}", f"{std_blinks:.2f}",
                     age, glasses, hours_sleep, caffeine, task, lighting,
                     score, map_self_label(score), f"baseline_blinks={baseline_blinks}"])

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("Session data saved to", CSV_FILE)
