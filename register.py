import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis


STUDENTS_FOLDER = "students"
TOTAL_SAMPLES = 10


def register_face(student_id, student_name):

    # ==========================================
    # CREATE STUDENT FOLDER
    # ==========================================

    student_folder = os.path.join(
        STUDENTS_FOLDER,
        student_id
    )

    os.makedirs(
        student_folder,
        exist_ok=True
    )

    # Remove old embeddings if they exist
    for filename in os.listdir(student_folder):

        if filename.endswith(".npy"):

            os.remove(
                os.path.join(
                    student_folder,
                    filename
                )
            )


    # ==========================================
    # LOAD INSIGHTFACE
    # ==========================================

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


    # ==========================================
    # OPEN CAMERA
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

        return False


    print("\n================================")
    print("FACE REGISTRATION")
    print("================================")
    print("ID   :", student_id)
    print("Name :", student_name)
    print()
    print("Keep only ONE face in the camera.")
    print("Press SPACE to capture.")
    print("Press Q to cancel.")
    print("================================")


    sample_number = 0


    # ==========================================
    # CAMERA LOOP
    # ==========================================

    while True:

        ret, frame = camera.read()

        if not ret:

            print("❌ Cannot read webcam")

            break


        faces = app.get(frame)


        # ======================================
        # DRAW FACE BOXES
        # ======================================

        for face in faces:

            box = face.bbox.astype(int)

            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face detected",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ======================================
        # DISPLAY INFORMATION
        # ======================================

        cv2.putText(
            frame,
            f"Student: {student_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_number}/{TOTAL_SAMPLES}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Capture | Q = Cancel",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        cv2.imshow(
            "AI Face Registration",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ======================================
        # CAPTURE SAMPLE
        # ======================================

        if key == 32:

            if len(faces) != 1:

                print(
                    "⚠️ Please make sure exactly "
                    "ONE face is visible."
                )

                continue


            face = faces[0]


            # Get face embedding
            embedding = face.embedding


            # Normalize embedding
            norm = np.linalg.norm(
                embedding
            )

            if norm > 0:

                embedding = (
                    embedding / norm
                )


            file_path = os.path.join(
                student_folder,
                f"embedding_{sample_number}.npy"
            )


            np.save(
                file_path,
                embedding
            )


            sample_number += 1


            print(
                f"✅ Sample "
                f"{sample_number}/{TOTAL_SAMPLES} saved"
            )


            # ==================================
            # REGISTRATION COMPLETE
            # ==================================

            if sample_number >= TOTAL_SAMPLES:

                print(
                    "\n🎉 Face registration completed!"
                )

                camera.release()

                cv2.destroyAllWindows()

                return True


        # ======================================
        # CANCEL
        # ======================================

        elif key == ord("q"):

            print(
                "\n❌ Face registration cancelled."
            )

            camera.release()

            cv2.destroyAllWindows()

            return False


    camera.release()

    cv2.destroyAllWindows()

    return False


# ==========================================
# DIRECT TERMINAL REGISTRATION
# ==========================================

if __name__ == "__main__":

    student_id = input(
        "Enter Student ID: "
    ).strip()

    student_name = input(
        "Enter Student Name: "
    ).strip()


    if not student_id or not student_name:

        print(
            "❌ Student ID and Name are required."
        )

        exit()


    register_face(
        student_id,
        student_name
    )