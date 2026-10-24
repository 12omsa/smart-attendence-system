import cv2
import os
import sqlite3
import threading
import time
from datetime import datetime

import numpy as np
from insightface.app import FaceAnalysis

from liveness import check_liveness


# ============================================================
# CONFIGURATION
# ============================================================

STUDENTS_FOLDER = "students"
DATABASE_FILE = "data/attendance.db"

# Face recognition threshold.
# A higher value is stricter.
RECOGNITION_THRESHOLD = 0.48

# Minimum face size for reliable processing.
MIN_FACE_SIZE = 70


# ============================================================
# SECURITY CONFIRMATION
# ============================================================

# Number of consecutive LIVE liveness checks required.
LIVE_CONFIRMATION_REQUIRED = 4

# Number of consecutive matching identity results required.
IDENTITY_CONFIRMATION_REQUIRED = 4

# Passive liveness is checked every AI cycle.
LIVENESS_CHECK_EVERY_CYCLE = 1


# ============================================================
# PERFORMANCE
# ============================================================

# Camera continues running at full rate.
# AI processing happens every 4th frame.
PROCESS_EVERY_N_FRAMES = 4

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
REQUESTED_CAMERA_FPS = 30

JPEG_QUALITY = 80


# ============================================================
# GLOBAL MODEL
# ============================================================

face_app = None


# ============================================================
# CAMERA READER
# ============================================================

class CameraReader:

    def __init__(self, camera_index=0):

        self.camera_index = camera_index

        self.camera = None

        self.latest_frame = None

        self.lock = threading.Lock()

        self.running = False

        self.thread = None

        self.error = None

    def open(self):

        self.camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():

            self.error = "Unable to open webcam."

            return False

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        self.camera.set(
            cv2.CAP_PROP_FPS,
            REQUESTED_CAMERA_FPS
        )

        self.camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

        return True

    def _capture_loop(self):

        while self.running:

            success, frame = self.camera.read()

            if not success:

                time.sleep(0.01)

                continue

            with self.lock:

                self.latest_frame = frame

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:

                return None

            return self.latest_frame.copy()

    def release(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

        if self.camera is not None:

            self.camera.release()

        self.camera = None


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def load_model():

    global face_app

    if face_app is None:

        print()
        print("==========================================")
        print("Loading InsightFace buffalo_s")
        print("==========================================")

        face_app = FaceAnalysis(

            name="buffalo_s",

            providers=[
                "CPUExecutionProvider"
            ]

        )

        face_app.prepare(

            ctx_id=0,

            det_size=(320, 320)

        )

        print("InsightFace loaded successfully.")
        print()

    return face_app


# ============================================================
# LOAD STUDENT EMBEDDINGS
# ============================================================

def load_student_embeddings():

    registered_students = {}

    if not os.path.exists(
        STUDENTS_FOLDER
    ):

        return registered_students

    for student_id in os.listdir(
        STUDENTS_FOLDER
    ):

        folder = os.path.join(
            STUDENTS_FOLDER,
            student_id
        )

        if not os.path.isdir(folder):

            continue

        embeddings = []

        for filename in os.listdir(folder):

            if not filename.endswith(".npy"):

                continue

            path = os.path.join(
                folder,
                filename
            )

            try:

                embedding = np.load(path)

                embedding = embedding.astype(
                    np.float32
                )

                norm = np.linalg.norm(
                    embedding
                )

                if norm <= 0:

                    continue

                embedding /= norm

                embeddings.append(
                    embedding
                )

            except Exception as error:

                print(
                    f"Could not load {path}:",
                    error
                )

        if not embeddings:

            continue

        average_embedding = np.mean(
            embeddings,
            axis=0
        )

        norm = np.linalg.norm(
            average_embedding
        )

        if norm <= 0:

            continue

        average_embedding /= norm

        registered_students[
            student_id
        ] = average_embedding

    print(
        "Registered embeddings:",
        len(registered_students)
    )

    return registered_students


# ============================================================
# LOAD STUDENT NAMES
# ============================================================

def load_student_names():

    names = {}

    if not os.path.exists(
        DATABASE_FILE
    ):

        return names

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT student_id, name
        FROM students
        """
    )

    rows = cursor.fetchall()

    for student_id, name in rows:

        names[student_id] = name

    connection.close()

    return names


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(
    face,
    registered_students
):

    if face.embedding is None:

        return None, 0.0

    embedding = face.embedding.astype(
        np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm <= 0:

        return None, 0.0

    embedding /= norm

    best_id = None
    best_score = -1.0

    for student_id, stored_embedding in (
        registered_students.items()
    ):

        score = float(
            np.dot(
                embedding,
                stored_embedding
            )
        )

        if score > best_score:

            best_score = score
            best_id = student_id

    if (

        best_id is not None

        and

        best_score >=
        RECOGNITION_THRESHOLD

    ):

        return best_id, best_score

    return None, best_score


# ============================================================
# ATTENDANCE
# ============================================================

def mark_attendance(student_id):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    cursor.execute(
        """
        SELECT id
        FROM attendance
        WHERE student_id = ?
        AND date = ?
        AND status = 'Present'
        """,
        (
            student_id,
            today
        )
    )

    existing = cursor.fetchone()

    if existing:

        connection.close()

        return False

    cursor.execute(
        """
        INSERT INTO attendance
        (
            student_id,
            date,
            time,
            status
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            student_id,
            today,
            current_time,
            "Present"
        )
    )

    connection.commit()

    connection.close()

    print(
        f"ATTENDANCE MARKED: {student_id}"
    )

    return True


# ============================================================
# RESULT OBJECT
# ============================================================

def make_result(
    box,
    student_id=None,
    name="",
    face_score=0.0,
    live_score=0.0,
    status="",
    color=(0, 255, 255)
):

    return {

        "box": box,

        "student_id": student_id,

        "name": name,

        "face_score": face_score,

        "live_score": live_score,

        "status": status,

        "color": color

    }


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(
    frame,
    result
):

    x1, y1, x2, y2 = result["box"]

    color = result["color"]

    student_id = result["student_id"]

    name = result["name"]

    face_score = result["face_score"]

    live_score = result["live_score"]

    status = result["status"]


    # ========================================================
    # FACE BOX
    # ========================================================

    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        color,

        3

    )


    # ========================================================
    # NAME / STATUS
    # ========================================================

    if student_id is not None:

        title = name

    else:

        title = status


    cv2.putText(

        frame,

        title,

        (
            x1,
            max(y1 - 12, 25)
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        color,

        2

    )


    # ========================================================
    # LIVE SCORE
    # ========================================================

    cv2.putText(

        frame,

        f"Liveness: {live_score:.2f}",

        (
            x1,
            min(
                y2 + 22,
                frame.shape[0] - 45
            )
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        color,

        2

    )


    # ========================================================
    # FACE SCORE
    # ========================================================

    if student_id is not None:

        cv2.putText(

            frame,

            f"Match: {face_score:.2f}",

            (
                x1,
                min(
                    y2 + 42,
                    frame.shape[0] - 20
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            color,

            2

        )


# ============================================================
# GENERATE FRAMES
# ============================================================

def generate_frames():

    model = load_model()

    registered_students = (
        load_student_embeddings()
    )

    student_names = (
        load_student_names()
    )


    # ========================================================
    # CAMERA
    # ========================================================

    camera = CameraReader(
        camera_index=0
    )

    if not camera.open():

        print(camera.error)

        return


    # ========================================================
    # STATE
    # ========================================================

    frame_count = 0

    ai_cycle = 0

    results = []

    person_states = {}

    attendance_marked = set()


    # ========================================================
    # FPS
    # ========================================================

    fps_counter = 0

    fps_start = time.perf_counter()

    displayed_fps = 0.0


    try:

        while True:

            frame = camera.get_frame()

            if frame is None:

                time.sleep(0.005)

                continue


            frame_count += 1


            # =================================================
            # AI PROCESSING
            # =================================================

            if (

                frame_count
                %
                PROCESS_EVERY_N_FRAMES
                ==
                0

            ):

                ai_cycle += 1

                # ---------------------------------------------
                # Detection frame
                # ---------------------------------------------

                detection_frame = cv2.resize(

                    frame,

                    (320, 240),

                    interpolation=cv2.INTER_LINEAR

                )


                faces = model.get(
                    detection_frame
                )


                scale_x = (
                    frame.shape[1] /
                    detection_frame.shape[1]
                )

                scale_y = (
                    frame.shape[0] /
                    detection_frame.shape[0]
                )


                new_results = []

                visible_ids = set()


                # =================================================
                # PROCESS FACES
                # =================================================

                for face in faces:

                    bbox = face.bbox.astype(
                        np.int32
                    )

                    x1 = int(
                        bbox[0] * scale_x
                    )

                    y1 = int(
                        bbox[1] * scale_y
                    )

                    x2 = int(
                        bbox[2] * scale_x
                    )

                    y2 = int(
                        bbox[3] * scale_y
                    )


                    x1 = max(
                        0,
                        min(
                            x1,
                            frame.shape[1] - 1
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            y1,
                            frame.shape[0] - 1
                        )
                    )

                    x2 = max(
                        0,
                        min(
                            x2,
                            frame.shape[1] - 1
                        )
                    )

                    y2 = max(
                        0,
                        min(
                            y2,
                            frame.shape[0] - 1
                        )
                    )


                    face_width = x2 - x1
                    face_height = y2 - y1


                    # =================================================
                    # FACE TOO SMALL
                    # =================================================

                    if (

                        face_width < MIN_FACE_SIZE

                        or

                        face_height < MIN_FACE_SIZE

                    ):

                        continue


                    full_bbox = (
                        x1,
                        y1,
                        x2,
                        y2
                    )


                    # =================================================
                    # FIRST SECURITY GATE:
                    # PASSIVE LIVENESS
                    # =================================================

                    liveness = check_liveness(

                        frame,

                        full_bbox

                    )


                    live_score = float(

                        liveness.get(
                            "real_score",
                            0.0
                        )

                    )


                    # =================================================
                    # SPOOF
                    # =================================================

                    if not liveness["is_real"]:

                        # ---------------------------------------------
                        # Do NOT recognize this face.
                        # ---------------------------------------------

                        new_results.append(

                            make_result(

                                full_bbox,

                                None,

                                "",

                                0.0,

                                live_score,

                                "SPOOF - ACCESS DENIED",

                                (0, 0, 255)

                            )

                        )

                        continue


                    # =================================================
                    # LIVENESS PASSED
                    # =================================================

                    # ---------------------------------------------
                    # Now and ONLY now perform identity matching.
                    # ---------------------------------------------

                    student_id, face_score = (
                        recognize_face(
                            face,
                            registered_students
                        )
                    )


                    # =================================================
                    # UNKNOWN LIVE PERSON
                    # =================================================

                    if student_id is None:

                        new_results.append(

                            make_result(

                                full_bbox,

                                None,

                                "",

                                face_score,

                                live_score,

                                "LIVE - UNKNOWN",

                                (0, 165, 255)

                            )

                        )

                        continue


                    visible_ids.add(
                        student_id
                    )


                    # =================================================
                    # CREATE STATE
                    # =================================================

                    if student_id not in person_states:

                        person_states[
                            student_id
                        ] = {

                            "live_frames": 0,

                            "identity_frames": 0,

                            "last_seen": ai_cycle,

                            "live_verified": False,

                            "identity_verified": False

                        }


                    state = person_states[
                        student_id
                    ]


                    state["last_seen"] = ai_cycle


                    # =================================================
                    # LIVE CONFIRMATION
                    # =================================================

                    state["live_frames"] += 1


                    if (

                        state["live_frames"]

                        >=

                        LIVE_CONFIRMATION_REQUIRED

                    ):

                        state["live_verified"] = True


                    # =================================================
                    # IDENTITY CONFIRMATION
                    # =================================================

                    if state["live_verified"]:

                        state[
                            "identity_frames"
                        ] += 1


                    if (

                        state["identity_frames"]

                        >=

                        IDENTITY_CONFIRMATION_REQUIRED

                    ):

                        state[
                            "identity_verified"
                        ] = True


                    # =================================================
                    # NAME
                    # =================================================

                    name = student_names.get(

                        student_id,

                        student_id

                    )


                    # =================================================
                    # FINAL VERIFIED STATE
                    # =================================================

                    if (

                        state["live_verified"]

                        and

                        state["identity_verified"]

                    ):

                        status = (
                            "VERIFIED - LIVE"
                        )

                        color = (
                            0,
                            255,
                            0
                        )


                        # ---------------------------------------------
                        # Attendance only happens here.
                        # ---------------------------------------------

                        if (

                            student_id
                            not in
                            attendance_marked

                        ):

                            mark_attendance(
                                student_id
                            )

                            attendance_marked.add(
                                student_id
                            )


                    elif state["live_verified"]:

                        status = (
                            "LIVE - VERIFYING ID"
                        )

                        color = (
                            0,
                            255,
                            255
                        )

                    else:

                        status = (
                            "LIVE CHECKING"
                        )

                        color = (
                            0,
                            255,
                            255
                        )


                    # =================================================
                    # RESULT
                    # =================================================

                    new_results.append(

                        make_result(

                            full_bbox,

                            student_id,

                            name,

                            face_score,

                            live_score,

                            status,

                            color

                        )

                    )


                # =================================================
                # CLEAN OLD STATES
                # =================================================

                for student_id in list(
                    person_states.keys()
                ):

                    state = person_states[
                        student_id
                    ]

                    if (

                        ai_cycle
                        -
                        state["last_seen"]

                        >

                        10

                    ):

                        del person_states[
                            student_id
                        ]


                results = new_results


            # ========================================================
            # DRAW CURRENT RESULTS
            # ========================================================

            for result in results:

                draw_result(
                    frame,
                    result
                )


            # ========================================================
            # FPS
            # ========================================================

            fps_counter += 1

            elapsed = (
                time.perf_counter()
                -
                fps_start
            )

            if elapsed >= 1.0:

                displayed_fps = (
                    fps_counter /
                    elapsed
                )

                fps_counter = 0

                fps_start = (
                    time.perf_counter()
                )


            # ========================================================
            # HEADER
            # ========================================================

            cv2.putText(

                frame,

                "AI ATTENDANCE",

                (20, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.70,

                (255, 255, 255),

                2

            )


            cv2.putText(

                frame,

                "PASSIVE ANTI-SPOOFING ACTIVE",

                (20, 55),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (0, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f"Camera FPS: {displayed_fps:.1f}",

                (20, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (0, 255, 0),

                2

            )


            # ========================================================
            # JPEG
            # ========================================================

            success, buffer = cv2.imencode(

                ".jpg",

                frame,

                [

                    cv2.IMWRITE_JPEG_QUALITY,

                    JPEG_QUALITY

                ]

            )


            if not success:

                continue


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                +

                buffer.tobytes()

                +

                b"\r\n"

            )


    finally:

        camera.release()

        cv2.destroyAllWindows()