import cv2
import os
import csv
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# ==========================================
# SETTINGS
# ==========================================

STUDENTS_FOLDER = "students"
DATA_FOLDER = "data"
ATTENDANCE_FILE = os.path.join(DATA_FOLDER, "attendance.csv")

# Recognition threshold
THRESHOLD = 0.45

# ==========================================
# CREATE DATA FOLDER
# ==========================================

os.makedirs(DATA_FOLDER, exist_ok=True)

# Create attendance CSV if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student ID",
            "Date",
            "Time",
            "Status"
        ])

# ==========================================
# LOAD REGISTERED STUDENTS
# ==========================================

student_embeddings = []
student_ids = []

print("Loading registered students...")

for student_id in os.listdir(STUDENTS_FOLDER):

    student_folder = os.path.join(
        STUDENTS_FOLDER,
        student_id
    )

    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    for filename in os.listdir(student_folder):

        if filename.endswith(".npy"):

            path = os.path.join(
                student_folder,
                filename
            )

            embedding = np.load(path)

            embedding = (
                embedding /
                np.linalg.norm(embedding)
            )

            embeddings.append(embedding)

    if embeddings:

        # Average embeddings
        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        mean_embedding = (
            mean_embedding /
            np.linalg.norm(mean_embedding)
        )

        student_embeddings.append(
            mean_embedding
        )

        student_ids.append(
            student_id
        )

        print(
            f"✅ Loaded {student_id}"
        )

if not student_embeddings:

    print("❌ No students registered.")
    exit()

# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("\nLoading AI model...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("✅ AI model loaded")

# ==========================================
# ATTENDANCE FUNCTION
# ==========================================

def mark_attendance(student_id):

    today = datetime.now().strftime("%Y-%m-%d")

    # Check if already marked today
    with open(
        ATTENDANCE_FILE,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["Student ID"] == student_id
                and row["Date"] == today
            ):

                return False

    # Mark attendance
    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            student_id,
            date,
            time,
            "Present"
        ])

    print(
        f"✅ Attendance marked: {student_id}"
    )

    return True


# ==========================================
# START CAMERA
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

if not camera.isOpened():

    print("❌ Cannot open webcam")
    exit()

print("\n================================")
print("AI ATTENDANCE SYSTEM")
print("================================")
print("Look at the camera.")
print("Press Q to quit.")
print("================================")

# ==========================================
# CAMERA LOOP
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        box = face.bbox.astype(int)

        x1, y1, x2, y2 = box

        # Get face embedding
        embedding = face.embedding

        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )

        # ----------------------------------
        # FIND BEST MATCH
        # ----------------------------------

        best_student = "Unknown"
        best_score = -1

        for i, registered_embedding in enumerate(
            student_embeddings
        ):

            score = np.dot(
                embedding,
                registered_embedding
            )

            if score > best_score:

                best_score = score
                best_student = student_ids[i]

        # ----------------------------------
        # CHECK RECOGNITION
        # ----------------------------------

        if best_score >= THRESHOLD:

            name = best_student

            # Mark attendance
            mark_attendance(
                best_student
            )

        else:

            name = "Unknown"

        # ----------------------------------
        # DRAW FACE
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{name} {best_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Registered Students: {len(student_ids)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "AI Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("\nAttendance system stopped.")