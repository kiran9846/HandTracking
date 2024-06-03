import cv2
import numpy as np

import Module

sequence = []
sentence = []
threshold = 0.4

cap = cv2.VideoCapture(0)
with Module.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = Module.mediapipe_detection(frame, holistic)
            print(results)

            Module.draw_styled_landmarks(image, results)  # draw landmarks

            keypoints = Module.extract_keypoints(results)
            sequence.insert(0,keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                res = Module.model.predict(np.expand_dims(sequence, axis=0))[0]
                print(Module.actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if Module.actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(Module.actions[np.argmax(res)])
                    else:
                        sentence.append(Module.actions[np.argmax(res)])




            cv2.imshow('OpenCV Feed', image)  # show to screen by passing to the image

            # if you hit the q it will break the code
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Program interrupted by the user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()