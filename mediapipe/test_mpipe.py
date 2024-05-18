import cv2
import mediapipe as mp
import csv
import numpy as np


mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

path = r'/home/chenzy/FastInst-main/手勢_裁切影片/手勢_原始影片/反手切/IMG_7408_Trim.mp4'
cap = cv2.VideoCapture(path)
fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('/home/chenzy/FastInst-main/mediapipe/IMG_7408_Trim_mediaP.mp4', codec, fps, (int(width/2), height),isColor=True)
print(fps, width, height)
lndmark_list = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while True:
        ret, img = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        else:
            # print(img.shape)
            img = img[0:1080, 960:1920]
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = pose.process(img2)                  # 取得姿勢偵測結果
                
                # 根據姿勢偵測結果，標記身體節點和骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            

            lndmark_list.append ([[results.pose_landmarks.landmark[i.value].x, \
                                      results.pose_landmarks.landmark[i.value].y, \
                                      results.pose_landmarks.landmark[i.value].z] for i in mp_pose.PoseLandmark])
            
            # print(results.pose_landmarks)
            # path_csv_str = "/home/chenzy/FastInst-main/mediapipe/IMG_7386_Trim_mediaP.txt"

            # with open(path_csv_str, 'w', newline='') as csvfile_1:
            #         writer = csv.writer(csvfile_1)
            #         # writer.writerow(['right', 'area'])
            #         writer.writerow([[results.pose_landmarks.landmark[i.value].x, \
            #                           results.pose_landmarks.landmark[i.value].y, \
            #                           results.pose_landmarks.landmark[i.value].z] for i in mp_pose.PoseLandmark])
                    
            output_video.write(img)
    np.save('/home/chenzy/FastInst-main/mediapipe/IMG_7408_Trim_mediaP.npy', lndmark_list)
    # print((lndmark_list[0][1].x))
    cap.release()
    output_video.release()
    # csvfile_1.close()