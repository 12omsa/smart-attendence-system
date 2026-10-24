import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

STUDENTS_FOLDER = "students"

# -----------------------------
# Load registered students
# -----------------------------

student_embeddings = []
student_names = []

for student_id in os.listdir(STUDENTS_FOLDER):

    student_folder = os.path.join(
        STUDENTS_FOLDER,
        student_id
    )

    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    for file in os.listdir(student_folder):

        if file.endswith(".npy"):

            path = os.path.join(
                student_folder,
                file
            )

            embedding = np.load(path)

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            embeddings.append(embedding)

    if embeddings:

        # Average all 10 samples
        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        # Normalize again
        mean_embedding = (
            mean_embedding /
            np.linalg.norm(mean_embedding)
        )

        student_embeddings.append(mean_embedding)
        student_names.append(student_id)

        print(
            f"✅ Loaded {student_id} "
            f"({len(embeddings)} samples)"
        )


if not student_embeddings:

    print("❌ No registered students found.")
    exit()


# -----------------------------
# Load InsightFace
# -----------------------------

print("\nLoading AI model...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("✅ Model loaded")


# -----------------------------
# Start camera
# -----------------------------

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
print("FACE RECOGNITION")
print("================================")
print("Press Q to quit")
print("================================")


while True:

    ret, frame = camera.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        # Face bounding box
        box = face.bbox.astype(int)

        x1, y1, x2, y2 = box

        # Current face embedding
        current_embedding = face.embedding

        current_embedding = (
            current_embedding /
            np.linalg.norm(current_embedding)
        )

        # -----------------------------
        # Compare with students
        # -----------------------------

        best_student = "Unknown"
        best_score = -1

        for i, registered_embedding in enumerate(
            student_embeddings
        ):

            score = np.dot(
                current_embedding,
                registered_embedding
            )

            if score > best_score:

                best_score = score
                best_student = student_names[i]

        # -----------------------------
        # Recognition threshold
        # -----------------------------

        if best_score >= 0.45:

            name = best_student

        else:

            name = "Unknown"

        # -----------------------------
        # Draw result
        # -----------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "AI Attendance Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
