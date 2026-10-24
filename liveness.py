import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FOLDER = "liveness_models"

INPUT_SIZE = (80, 80)

# Minimum confidence required for a real prediction
REAL_CONFIDENCE_THRESHOLD = 0.75

# Model labels used by Silent-Face-Anti-Spoofing
LABEL_NAMES = [
    "Paper Photo",
    "Real Face",
    "Screen Photo"
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = [
    {
        "filename": "2.7_80x80_MiniFASNetV2.onnx",
        "scale": 2.7
    },
    {
        "filename": "4_0_0_80x80_MiniFASNetV1SE.onnx",
        "scale": 4.0
    }
]


# ============================================================
# LIVENESS DETECTOR
# ============================================================

class LivenessDetector:

    def __init__(self):

        self.sessions = []

        self.load_models()


    # ========================================================
    # LOAD MODELS
    # ========================================================

    def load_models(self):

        model_folder = Path(
            MODEL_FOLDER
        )

        if not model_folder.exists():

            raise FileNotFoundError(
                f"Liveness model folder not found: "
                f"{MODEL_FOLDER}"
            )


        print()
        print("==========================================")
        print("Loading Anti-Spoofing Models")
        print("==========================================")


        for model_config in MODEL_CONFIG:

            model_path = (
                model_folder
                /
                model_config["filename"]
            )


            if not model_path.exists():

                raise FileNotFoundError(
                    f"Anti-spoofing model not found: "
                    f"{model_path}"
                )


            try:

                session = ort.InferenceSession(
                    str(model_path),
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )


                input_name = (
                    session
                    .get_inputs()[0]
                    .name
                )


                output_name = (
                    session
                    .get_outputs()[0]
                    .name
                )


                self.sessions.append({

                    "session":
                        session,

                    "input_name":
                        input_name,

                    "output_name":
                        output_name,

                    "scale":
                        model_config["scale"],

                    "filename":
                        model_config["filename"]

                })


                print(
                    "Loaded:",
                    model_config["filename"]
                )


            except Exception as error:

                raise RuntimeError(
                    "Unable to load anti-spoofing "
                    f"model {model_path}: {error}"
                )


        if not self.sessions:

            raise RuntimeError(
                "No anti-spoofing models loaded."
            )


        print(
            f"Anti-spoofing models loaded: "
            f"{len(self.sessions)}"
        )

        print("==========================================")
        print()


    # ========================================================
    # CALCULATE EXPANDED FACE BOX
    # ========================================================

    @staticmethod
    def get_expanded_box(
        frame_width,
        frame_height,
        bbox,
        scale
    ):

        x1, y1, x2, y2 = bbox

        box_width = max(
            x2 - x1,
            1
        )

        box_height = max(
            y2 - y1,
            1
        )


        scale = min(

            (frame_height - 1)
            / box_height,

            (frame_width - 1)
            / box_width,

            scale

        )


        new_width = (
            box_width * scale
        )

        new_height = (
            box_height * scale
        )


        center_x = (
            x1 + box_width / 2
        )

        center_y = (
            y1 + box_height / 2
        )


        left = (
            center_x
            -
            new_width / 2
        )

        top = (
            center_y
            -
            new_height / 2
        )

        right = (
            center_x
            +
            new_width / 2
        )

        bottom = (
            center_y
            +
            new_height / 2
        )


        # Keep crop inside frame

        if left < 0:

            right -= left
            left = 0


        if top < 0:

            bottom -= top
            top = 0


        if right > frame_width - 1:

            left -= (
                right
                -
                frame_width
                +
                1
            )

            right = (
                frame_width - 1
            )


        if bottom > frame_height - 1:

            top -= (
                bottom
                -
                frame_height
                +
                1
            )

            bottom = (
                frame_height - 1
            )


        left = max(
            0,
            int(left)
        )

        top = max(
            0,
            int(top)
        )

        right = min(
            frame_width - 1,
            int(right)
        )

        bottom = min(
            frame_height - 1,
            int(bottom)
        )


        return (
            left,
            top,
            right,
            bottom
        )


    # ========================================================
    # PREPARE FACE
    # ========================================================

    def preprocess_face(
        self,
        frame,
        bbox,
        scale
    ):

        frame_height, frame_width = (
            frame.shape[:2]
        )


        left, top, right, bottom = (
            self.get_expanded_box(
                frame_width,
                frame_height,
                bbox,
                scale
            )
        )


        if right <= left or bottom <= top:

            return None


        face_crop = frame[
            top:bottom + 1,
            left:right + 1
        ]


        if face_crop.size == 0:

            return None


        face_resized = cv2.resize(
            face_crop,
            INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR
        )


        # IMPORTANT:
        # MiniFASNet expects BGR input.
        face_float = (
            face_resized
            .astype(np.float32)
        )


        # HWC -> CHW

        face_chw = np.transpose(
            face_float,
            (2, 0, 1)
        )


        # Add batch dimension

        face_batch = np.expand_dims(
            face_chw,
            axis=0
        )


        return face_batch


    # ========================================================
    # SOFTMAX
    # ========================================================

    @staticmethod
    def softmax(values):

        values = (
            values
            -
            np.max(
                values,
                axis=1,
                keepdims=True
            )
        )


        exponential = np.exp(
            values
        )


        return (
            exponential
            /
            np.sum(
                exponential,
                axis=1,
                keepdims=True
            )
        )


    # ========================================================
    # PREDICT LIVENESS
    # ========================================================

    def predict(
        self,
        frame,
        bbox
    ):

        all_predictions = []


        for model in self.sessions:

            input_data = (
                self.preprocess_face(
                    frame,
                    bbox,
                    model["scale"]
                )
            )


            if input_data is None:

                return {

                    "is_real": False,

                    "label": "Invalid Face Crop",

                    "confidence": 0.0,

                    "real_score": 0.0,

                    "paper_score": 0.0,

                    "screen_score": 0.0

                }


            output = (
                model["session"].run(
                    [
                        model["output_name"]
                    ],
                    {
                        model["input_name"]:
                            input_data
                    }
                )[0]
            )


            probabilities = (
                self.softmax(output)
            )


            probabilities = (
                probabilities[0]
            )


            all_predictions.append(
                probabilities
            )


        # Average predictions from both models

        average_prediction = np.mean(
            np.asarray(
                all_predictions
            ),
            axis=0
        )


        predicted_class = int(
            np.argmax(
                average_prediction
            )
        )


        paper_score = float(
            average_prediction[0]
        )

        real_score = float(
            average_prediction[1]
        )

        screen_score = float(
            average_prediction[2]
        )


        # A face is accepted only when:
        #
        # 1. REAL is the strongest class
        # 2. REAL confidence reaches threshold

        is_real = (

            predicted_class == 1

            and

            real_score
            >=
            REAL_CONFIDENCE_THRESHOLD

        )


        label = LABEL_NAMES[
            predicted_class
        ]


        return {

            "is_real":
                bool(is_real),

            "label":
                label,

            "confidence":
                float(
                    average_prediction[
                        predicted_class
                    ]
                ),

            "real_score":
                real_score,

            "paper_score":
                paper_score,

            "screen_score":
                screen_score

        }


# ============================================================
# GLOBAL DETECTOR
# ============================================================

_liveness_detector = None


def load_liveness_model():

    global _liveness_detector


    if _liveness_detector is None:

        _liveness_detector = (
            LivenessDetector()
        )


    return _liveness_detector


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def check_liveness(
    frame,
    bbox
):

    detector = (
        load_liveness_model()
    )


    return detector.predict(
        frame,
        bbox
    )