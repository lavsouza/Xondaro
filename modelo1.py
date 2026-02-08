import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pose_result = None
hand_result = None
face_result = None

def face_callback(result, output_image, timestamp_ms):
    global face_result
    face_result = result

def pose_callback(result, output_image, timestamp_ms):
    global pose_result
    pose_result = result

def hand_callback(result, output_image, timestamp_ms):
    global hand_result
    hand_result = result


face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='face_landmarker.task'
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    result_callback=face_callback
)

face_detector = vision.FaceLandmarker.create_from_options(face_options)


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
FACE_CONNECTIONS = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE

def draw_all(image, pose_result, hand_result, face_result):
    imagem_capturada = image.copy()
    h, w, _ = imagem_capturada.shape

    # =============================
    # POSE (CORPO)
    # =============================
    if pose_result and pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]

        # Conexões do corpo
        for connection in POSE_CONNECTIONS:
            start = landmarks[connection.start]
            end = landmarks[connection.end]

            cv2.line(
                imagem_capturada,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (255, 0, 0), 2
            )

        # Pontos do corpo
        for lm in landmarks:
            cv2.circle(
                imagem_capturada,
                (int(lm.x * w), int(lm.y * h)),
                5,
                (0, 255, 255), -1
            )

    # =============================
    # MÃOS
    # =============================
    if hand_result and hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:

            # Conexões da mão
            for connection in HAND_CONNECTIONS:
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                cv2.line(
                    imagem_capturada,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (0, 255, 0), 2
                )

            # Pontos da mão
            for lm in hand_landmarks:
                cv2.circle(
                    imagem_capturada,
                    (int(lm.x * w), int(lm.y * h)),
                    4,
                    (0, 0, 255), -1
                )

    # =============================
    # FACE
    # =============================
    if face_result and face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]

        for lm in face_landmarks:
            cv2.circle(
                imagem_capturada,
                (int(lm.x * w), int(lm.y * h)),
                1,
                (255, 255, 255), -1
            )

    return imagem_capturada

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

    # pose_detector.detect_async(mp_image, timestamp)
    hand_detector.detect_async(mp_image, timestamp)
    # face_detector.detect_async(mp_image, timestamp)

    imagem_exibida = draw_all(
        frame_bgr,
        pose_result,
        hand_result,
        face_result
    )

    largura_exibicao = 1280
    altura_exibicao = 720
    janela_maior = cv2.resize(imagem_exibida, (largura_exibicao, altura_exibicao))

    cv2.imshow("Captura Holistica (Corpo, Maos e Rosto)", janela_maior)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()