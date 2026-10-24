import cv2
import time

from liveness import load_liveness_model


CAMERA_INDEX = 0

WINDOW_NAME = "Liveness Test"

detector = load_liveness_model()

camera = cv2.VideoCapture(
    CAMERA_INDEX,
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
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)


print()
print("==============================================")
print("        DIRECT LIVENESS MODEL TEST")
print("==============================================")
print()
print("Test 1: Look at the camera normally.")
print("Test 2: Hold a recorded video of your face")
print("        in front of the webcam.")
print("Test 3: Hold a photo of your face.")
print()
print("Press Q to quit.")
print("==============================================")
print()


last_test_time = 0

last_result = {
    "label": "Waiting...",
    "real_score": 0.0,
    "paper_score": 0.0,
    "screen_score": 0.0,
    "is_real": False,
    "confidence": 0.0
}


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break


    height, width = frame.shape[:2]


    # ----------------------------------------------------------
    # Detect face using the same InsightFace model
    # ----------------------------------------------------------

    from recognition import load_model

    face_model = load_model()

    faces = face_model.get(frame)


    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        # Use the largest detected face
        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0])
            *
            (f.bbox[3] - f.bbox[1])
        )

        bbox = face.bbox.astype(int)

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width - 1, int(bbox[2]))
        y2 = min(height - 1, int(bbox[3]))


        # ------------------------------------------------------
        # Run MiniFASNet
        # ------------------------------------------------------

        current_time = time.time()

        # Don't run the model on every camera frame.
        if current_time - last_test_time >= 0.15:

            try:

                last_result = detector.predict(
                    frame,
                    (x1, y1, x2, y2)
                )

            except Exception as error:

                print(
                    "Liveness error:",
                    error
                )

                last_result = {
                    "label": "ERROR",
                    "real_score": 0.0,
                    "paper_score": 0.0,
                    "screen_score": 0.0,
                    "is_real": False,
                    "confidence": 0.0
                }

            last_test_time = current_time


        # ------------------------------------------------------
        # Result
        # ------------------------------------------------------

        if last_result["is_real"]:

            box_color = (
                0,
                255,
                0
            )

            result_text = "REAL"

        else:

            box_color = (
                0,
                0,
                255
            )

            result_text = "SPOOF"


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )


        # ------------------------------------------------------
        # Display scores
        # ------------------------------------------------------

        cv2.putText(
            frame,
            result_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            box_color,
            3
        )


        cv2.putText(
            frame,
            f"Model: {last_result['label']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"REAL: {last_result['real_score']:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"PAPER: {last_result['paper_score']:.3f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"SCREEN: {last_result['screen_score']:.3f}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"CONFIDENCE: {last_result['confidence']:.3f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()

print()
print("Liveness test finished.")