import cv2
import mediapipe as mp
import pyautogui  # Biblioteca para controlar el teclado y multimedia

# Inicialización de MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Variables para almacenar la posición inicial de los dedos y control de modos
previous_heights = [None, None]  # Guardar las posiciones anteriores de las dos manos
gesture_active = False

# Función para verificar si los dedos índice, medio y anular están extendidos
def are_fingers_extended(landmarks):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    
    return (index_tip.y < index_dip.y and middle_tip.y < middle_dip.y and ring_tip.y < ring_dip.y)

# Función para detectar el deslizamiento de los dedos
def detect_swipe_direction(start_pos, current_pos):
    x_movement = current_pos[0] - start_pos[0]
    if x_movement < -0.2:  # Movimiento hacia la derecha (inversión por perspectiva)
        return "right"
    elif x_movement > 0.2:  # Movimiento hacia la izquierda
        return "left"
    return None

# Función para calcular la altura promedio de la mano (usaremos el punto de la muñeca)
def calculate_hand_height(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return wrist.y

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb_frame)

    if result_hands.multi_hand_landmarks and len(result_hands.multi_hand_landmarks) == 2:
        # Modo de dos manos: control de volumen
        for i, hand_landmarks in enumerate(result_hands.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener la altura de cada mano
            current_height = calculate_hand_height(hand_landmarks.landmark)

            # Verificar si tenemos una posición anterior para comparar
            if previous_heights[i] is not None:
                if current_height < previous_heights[i] - 0.02:  # Manos suben
                    print("Subiendo volumen")
                    pyautogui.press("volumeup")  # Subir volumen
                elif current_height > previous_heights[i] + 0.02:  # Manos bajan
                    print("Bajando volumen")
                    pyautogui.press("volumedown")  # Bajar volumen

            # Actualizar la altura anterior para la próxima iteración
            previous_heights[i] = current_height

    elif result_hands.multi_hand_landmarks and len(result_hands.multi_hand_landmarks) == 1:
        # Modo de una mano: control de canciones
        hand_landmarks = result_hands.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Verificar si los tres dedos están extendidos
        if are_fingers_extended(hand_landmarks.landmark):
            if not gesture_active:
                start_position = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
                gesture_active = True
            else:
                current_position = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
                
                direction = detect_swipe_direction(start_position, current_position)
                if direction == "right":
                    print("Siguiente canción")
                    pyautogui.press("nexttrack")  # Siguiente canción
                    gesture_active = False
                elif direction == "left":
                    print("Canción anterior")
                    pyautogui.press("prevtrack")  # Canción anterior
                    gesture_active = False
        else:
            gesture_active = False  # Reiniciar si no están extendidos

    else:
        # Reiniciar alturas anteriores si no se detectan dos manos
        previous_heights = [None, None]

    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
