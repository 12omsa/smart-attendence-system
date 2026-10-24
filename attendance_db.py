import cv2
import os
import sqlite3
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis


# ==============================
# SETTINGS
# ==============================

STUDENTS_FOLDER = "students"
DATABASE_FILE = "data/attendance.db"

THRESHOLD = 0.45


# ==============================
# LOAD STUDENTS
# ==============================

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

            # Normalize
            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

    if embeddings:

        # Average all embeddings
        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        # Normalize again
        norm = np.linalg.norm(mean_embedding)

        if norm > 0:
            mean_embedding = (
                mean_embedding / norm
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

    print("❌ No registered students found.")

    exit()


# ==============================
# DATABASE FUNCTIONS
# ==============================

def get_student_name(student_id):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT name
        FROM students
        WHERE student_id = ?
        """,
        (student_id,)
    )

    result = cursor.fetchone()

    connection.close()

    if result:
        return result[0]

    return "Unknown"


def mark_attendance(student_id):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # Check if already marked
    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM attendance
        WHERE student_id = ?
        AND date = ?
        """,
        (student_id, today)
    )

    existing = cursor.fetchone()

    if existing:

        connection.close()

        return False


    # Get current time
    now = datetime.now()

    date = now.strftime(
        "%Y-%m-%d"
    )

    time = now.strftime(
        "%H:%M:%S"
    )


    # Insert attendance
    cursor.execute(
        """
        INSERT INTO attendance
        (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
        """,
        (
            student_id,
            date,
            time,
            "Present"
        )
    )

    connection.commit()

    connection.close()

    print(
        f"✅ Attendance marked: "
        f"{student_id}"
    )

    return True


# ==============================
# LOAD AI MODEL
# ==============================

print("\nLoading AI model...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=[
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("✅ AI model loaded")


# ==============================
# CAMERA
# ==============================

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


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = camera.read()

    if not ret:

        print(
            "❌ Cannot read webcam"
        )

        break


    faces = app.get(frame)


    for face in faces:

        box = face.bbox.astype(int)

        x1, y1, x2, y2 = box


        # Get face embedding
        embedding = face.embedding

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:

            embedding = (
                embedding / norm
            )


        # Find best match
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

                best_student = (
                    student_ids[i]
                )


        # ==========================
        # RECOGNIZED
        # ==========================

        if best_score >= THRESHOLD:

            student_id = best_student

            student_name = (
                get_student_name(
                    student_id
                )
            )

            mark_attendance(
                student_id
            )

            label = (
                f"{student_name} "
                f"{best_score:.2f}"
            )

        else:

            label = (
                f"Unknown "
                f"{best_score:.2f}"
            )


        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Draw name
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Display
    cv2.putText(
        frame,
        f"Students: {len(student_ids)}",
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


# ==============================
# CLEANUP
# ==============================

camera.release()

cv2.destroyAllWindows()

print(
    "\nAttendance system stopped."
)