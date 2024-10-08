import cv2
import mediapipe as mp

# Inicialización de MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Variables para almacenar la posición inicial de los dedos
start_position = None
gesture_active = False

def are_fingers_extended(landmarks):
    # Verificar si los dedos índice, medio y anular están extendidos en posición vertical
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    
    # Para que estén extendidos, los puntos TIP deben estar más arriba (menor valor y) que los DIP
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    
    return (index_tip.y < index_dip.y and middle_tip.y < middle_dip.y and ring_tip.y < ring_dip.y)

def detect_swipe_direction(start_pos, current_pos):
    # Detectar si los dedos se movieron hacia la derecha o izquierda
    x_movement = current_pos[0] - start_pos[0]
    
    # Invertir las direcciones por la perspectiva de la cámara
    if x_movement < -0.2:  # Ahora movimiento hacia la "derecha" será negativo
        return "right"
    elif x_movement > 0.2:  # Movimiento hacia la "izquierda" será positivo
        return "left"
    return None

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verificar si los tres dedos están extendidos
            if are_fingers_extended(hand_landmarks.landmark):
                if not gesture_active:
                    # Si el gesto recién empieza, guardamos la posición inicial
                    start_position = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
                    gesture_active = True
                else:
                    # Si el gesto está activo, detectamos el movimiento
                    current_position = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
                    
                    direction = detect_swipe_direction(start_position, current_position)
                    if direction == "right":
                        print("Gesto detectado: Siguiente canción")
                        gesture_active = False  # Reiniciar el gesto
                    elif direction == "left":
                        print("Gesto detectado: Canción anterior")
                        gesture_active = False  # Reiniciar el gesto
            else:
                gesture_active = False  # Reiniciar si no están extendidos

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
