import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic                   # mediapipe 偵測手掌方法

path = r'/home/chenzy/FastInst-main/new_record/left_player/right/fhpull_left_1_crop.mp4'
cap = cv2.VideoCapture(path)
fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('/home/chenzy/FastInst-main/mediapipe/new_record/left/fhpull_left_1_crop_holistic.mp4', codec, fps, (int(width), height),isColor=True)
# print(fps, width, height)
lndmark_list = []
# mediapipe 啟用偵測手掌
with mp_holistic.Holistic(
    # model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

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
            results = holistic.process(img2)                 
            if results.pose_landmarks:
                # mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                #              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                #              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                #              ) 
                lndmark_list.append ([ [results.pose_landmarks.landmark[i.value].x, \
                        results.pose_landmarks.landmark[i.value].y, \
                        results.pose_landmarks.landmark[i.value].z] for i in mp_holistic.PoseLandmark])
                # print([ [results.pose_landmarks.landmark[i.value].x, \
                #         results.pose_landmarks.landmark[i.value].y, \
                #         results.pose_landmarks.landmark[i.value].z] for i in mp_holistic.PoseLandmark])
                # # print([i for i in mp_holistic.PoseLandmark])
                

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
                output_video.write(img)
                lndmark_list.append ([[results.right_hand_landmarks.landmark[i].x, \
                                       results.right_hand_landmarks.landmark[i].y, \
                                       results.right_hand_landmarks.landmark[i].z] for i in mp_holistic.HandLandmark])
                # print(mp_holistic.HandLandmark.WRIST)
                
                # print([[results.right_hand_landmarks.landmark[i].x, \
                #                        results.right_hand_landmarks.landmark[i].y, \
                #                        results.right_hand_landmarks.landmark[i].z] for i in mp_holistic.HandLandmark])
                
                # break
                   
            else:
                np.zeros(21*3)
    # np.save('/home/chenzy/FastInst-main/mediapipe/bhc_left_1_hol.npy', lndmark_list)  
    # print((lndmark_list[0]))
    cap.release()
    output_video.release()
    # cv2.destroyAllWindows()