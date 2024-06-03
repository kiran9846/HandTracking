import cv2
import numpy as np
import os
import tensorflow
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.callbacks import TensorBoard
import results as results
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting a color from BRG 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


"""
def draw_landmarks(image , results):  # Draw a connections
    mp_drawing.draw_landmarks(image , results.face_landmarks , mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image , results.pose_landmarks , mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image , results.left_hand_landmarks , mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image , results.right_hand_landmarks , mp_holistic.HAND_CONNECTIONS)
"""


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2))

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255, 256, 121), thickness=2, circle_radius=2))

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 110, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255, 256, 121), thickness=2, circle_radius=2))


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
            print(results)

            draw_styled_landmarks(image, results)  # draw landmarks

            cv2.imshow('OpenCV Feed', image)  # show to screen by passing to the image

            # if you hit the q it will break the code
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Program interrupted by the user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

results
draw_styled_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # converting the landmarks into color


# appending the landmarks into an array
def extract_keyPoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


result_test = extract_keyPoints(results)
# saving the results into numpy '0' means the file where the result test will be save
np.save('0', result_test)
np.load('0.npy')

# path for extracting a data, numpy arrays
DATA_PATH = '/Users/kiranthapa/Desktop/SignLanguage/MP_Data'
actions = np.array(['Hello', 'ThankYou', 'IloveYou'])  # Actions that we try to detect
no_sequences = 30  # Thirty video worth of data
sequence_length = 30  # Videos are going to be 30 frame in length

# So in this loop we are creating a folder of action and making 30 different france
for action in actions:
    for sequence in range(no_sequences):
        try:
            folder_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(folder_path)
        except:
            pass

# creating a function for the collecting a frame
cap = cv2.VideoCapture(0)
# Set mideapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:
        # loop through sequence mean videos
        for sequence in range(no_sequences):
            # Loop through video length in sequence length
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()
                # Make detection
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                # draw landmarks
                draw_styled_landmarks(image, results)
                # Wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                (0, 255, 0), 5, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting Frame for {} video Number {}'.format(action, sequence),
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting Frame for {} video Number {}'.format(action, sequence),
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                # New export data
                keypoints = extract_keyPoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.distroyAllWindows()

from sklearn.model_selection import train_test_split

label_map = {label:num for num, label in enumerate(actions)}

sequence, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequence.append(window)
        labels.append(label_map[action])

x = np.array(sequence)
y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

#Building and Training LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

res = [.7,0.2,0.1]
actions[np.argmax(res)]
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train,y_train, epochs=2000, callbacks=[tb_callback])

#Making a prediction
res = model.predict(x_test)

#Save weights
model.save('action.h5')

#Evaluation using confusion Matrix and Accuracy
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(x_train)

ytrue = np.argmax(y_train,axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)




