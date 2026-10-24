import cv2
import time
from insightface.app import FaceAnalysis

print("Loading model...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("✅ Model loaded")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam started")
print("Press Q to quit")

faces = []

frame_count = 0
last_time = time.time()
fps = 0

while True:

    ret, frame = camera.read()

    if not ret:
        print("❌ Cannot read webcam")
        break

    frame_count += 1

    # Run AI detection every 5th frame
    if frame_count % 5 == 0:

        small_frame = cv2.resize(
            frame,
            (320, 240)
        )

        faces = app.get(small_frame)

    # Draw detected faces
    for face in faces:

        box = face.bbox.astype(int)

        x1 = int(box[0] * 2)
        y1 = int(box[1] * 2)
        x2 = int(box[2] * 2)
        y2 = int(box[3] * 2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # Calculate FPS
    current_time = time.time()

    if current_time - last_time >= 1:

        fps = frame_count / (current_time - last_time)

        frame_count = 0
        last_time = current_time

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "AI Attendance Camera",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()