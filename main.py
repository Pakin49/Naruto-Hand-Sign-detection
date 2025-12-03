import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
import keras

from PIL import Image
from rembg import remove


mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_hands = solutions.hands
cap = cv2.VideoCapture(0)
model = keras.saving.load_model("vgg19_naruto_hand_sign_model.h5")


def crop_hand(x_coords, y_coords, h, w, frame):
    # Calculate bounding box base on the dimensions of screen capture * x,y cooords
    x_min = int(min(x_coords) * w)
    x_max = int(max(x_coords) * w)
    y_min = int(min(y_coords) * h)
    y_max = int(max(y_coords) * h)

    x_len = x_max - x_min
    y_len = y_max - y_min

    # padding when size < 224
    padding_x = 0
    padding_y = 0
    if x_len <= 350:
        padding_x = int((224 - x_len) / 2)
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
    if y_len < 350:
        padding_y = int((224 - y_len) / 2)
        y_max = min(h, y_max + padding_y)
        y_min = max(0, y_min - padding_y)
    # Crop hand
    hand_crop = frame[y_min:y_max, x_min:x_max]

    # Draw box for visualization
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return hand_crop


def removebg_img(frame):
    pil_img = Image.fromarray(frame)

    img_nobg = remove(pil_img)  # background removed (RGBA)
    img_nobg = np.array(img_nobg)
    output = img_nobg[..., :3]
    return output


def get_prediciton(y_pred_proba):
    print(y_pred_proba)
    hand_sign = [
        "bird",
        "boar",
        "dog",
        "dragon",
        "hare",
        "horse",
        "monkey",
        "ox",
        "ram",
        "rat",
        "snake",
        "tiger",
        "idk",
    ]
    y_pred_index = -1
    predict_index = np.argmax(y_pred_proba)
    if y_pred_proba[0][predict_index] < 0.5:
        return hand_sign[y_pred_index]
    return hand_sign[-1]


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Detect one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
) as hands:

    frame_count = 0
    interval = 10
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get frame dimensions
                h, w, c = frame.shape

                # Extract all landmark coordinates
                # land mark coordinates are save as 0 to 1
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                hand_crop = crop_hand(x_coords, y_coords, h, w, frame)

                # print(hand_crop)
                if hand_crop is not None and hand_crop.size != 0:
                    #hand_crop = removebg_img(hand_crop)

                    cv2.imshow("crop", hand_crop)
                    cv2.moveWindow("crop", 300, 600)

                    hand_crop_resize = cv2.resize(hand_crop, (224, 224))

                    # normalized to 0-1
                    img = hand_crop_resize.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)

                    if frame_count % interval == 0:
                        prediction = model.predict(img)
                        predict = get_prediciton(prediction)
                        print(predict)
                        cv2.putText(
                            frame,
                            f"Prediction: {predict}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                frame_count += 1

                # Now hand_crop contains just the hand region
                # Next: resize to 224x224 and feed to model

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
