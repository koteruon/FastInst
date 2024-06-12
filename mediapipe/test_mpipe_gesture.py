import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

path = r'/home/chenzy/FastInst-main/new_record/left/fhpull_left_1.mp4'
cap = cv2.VideoCapture(path)
fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('/home/chenzy/FastInst-main/mediapipe/new_record/left/fhp_left_1_gesture.mp4', codec, fps, (int(width), height),isColor=True)
# print(fps, width, height)
lndmark_list = []
# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    # model_complexity=0,
    # max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        else:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = hands.process(img2)                 # 偵測手掌
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 將節點和骨架繪製到影像中
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    output_video.write(img)
                    # print(len(mp_hands))
                    # print(results.multi_hand_landmarks[0])
                    lndmark_list.append(results.multi_hand_landmarks[0].landmark)
                    # lndmark_list.append ([[results.multi_hand_landmarks[i.value].x, \
                    #                   results.multi_hand_landmarks[i.value].y, \
                    #                   results.multi_hand_landmarks[i.value].z] for i in results.multi_hand_landmarks])
            # mp_hands.Landmarks
            # cv2.imshow('oxxostudio', img)
            # if cv2.waitKey(5) == ord('q'):
            #     break    # 按下 q 鍵停止
            # output_video.write(img)
    np.save('/home/chenzy/FastInst-main/mediapipe/bhc_left_1.npy', lndmark_list)  
    print((lndmark_list[0]))
    cap.release()
    output_video.release()
    # cv2.destroyAllWindows()