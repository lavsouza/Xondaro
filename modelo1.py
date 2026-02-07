import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pose_result = None
hand_result = None

def pose_callback(result, output_image, timestamp_ms):
    global pose_result
    pose_result = result


def hand_callback(result, output_image, timestamp_ms):
    global hand_result
    hand_result = result


# =============================
# HAND LANDMARKER
# =============================
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='hand_landmarker.task'
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=4,
    result_callback=hand_callback
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# =============================
# POSE LANDMARKER
# =============================
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='pose_landmarker_full.task'
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=pose_callback
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

def draw_pose_and_hands(image, pose_result, hand_result):
    annotated = image.copy()
    h, w, _ = annotated.shape

    # =============================
    # POSE
    # =============================
    if pose_result and pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]

        for connection in POSE_CONNECTIONS:
            start = landmarks[connection.start]
            end = landmarks[connection.end]

            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (255, 0, 0), 2
            )

        for lm in landmarks:
            cv2.circle(
                annotated,
                (int(lm.x * w), int(lm.y * h)),
                4, (255, 0, 0), -1
            )

    # =============================
    # MÃOS
    # =============================
    if hand_result and hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            for connection in HAND_CONNECTIONS:
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                cv2.line(
                    annotated,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (0, 255, 0), 2
                )

            for lm in hand_landmarks:
                cv2.circle(
                    annotated,
                    (int(lm.x * w), int(lm.y * h)),
                    4, (0, 0, 255), -1
                )

    return annotated

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a câmera")

timestamp = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp += 1

    pose_detector.detect_async(mp_image, timestamp)
    hand_detector.detect_async(mp_image, timestamp)

    annotated = draw_pose_and_hands(
        frame_bgr,
        pose_result,
        hand_result
    )

    cv2.imshow("Body + Hands - Live", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()