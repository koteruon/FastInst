import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import time

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic                   # mediapipe 偵測手掌方法

# path = r'/home/chenzy/FastInst-main/new_record/left_player/right/fhpull_left_1_crop.mp4'
# cap = cv2.VideoCapture(path)
# fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# codec = cv2.VideoWriter_fourcc(*'mp4v')
# output_video = cv2.VideoWriter('/home/chenzy/FastInst-main/mediapipe/new_record/left/fhpull_left_1_crop_holistic.mp4', codec, fps, (int(width), height),isColor=True)
# # print(fps, width, height)
# lndmark_list = []

def remove_landmark_connections(custom_connections, landmark):
    return filter(lambda con: is_landmark_in_connection(con, landmark), custom_connections)

def is_landmark_in_connection(connection, landmark):
    return landmark.value not in connection \
                  and not ( mp_holistic.PoseLandmark.LEFT_SHOULDER in connection and  mp_holistic.PoseLandmark.RIGHT_SHOULDER in connection)
def detect_right_hand(input_video, output_path):
# mediapipe 啟用偵測手掌
    start_time = time.time()
    path = input_video
    cap = cv2.VideoCapture(path)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path + ".mp4", codec, fps, (int(width), height),isColor=True)
    # print(fps, width, height)
    # lndmark_list = []
    proceed_frame = 0

    with mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.15,
        min_tracking_confidence=0.15) as holistic:

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            else:
                proceed_frame+=1
                # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
                # img2 = img[0:1080, 0:960]
                results = holistic.process(img)  
                custom_lm_styles = mp_drawing_styles.get_default_pose_landmarks_style() 
                custom_connections = list(mp_holistic.POSE_CONNECTIONS)
                included_landmarks = [
                    # right hand set
                    mp_holistic.PoseLandmark.RIGHT_SHOULDER, 
                    mp_holistic.PoseLandmark.RIGHT_ELBOW, 
                    mp_holistic.PoseLandmark.RIGHT_WRIST, 
                    mp_holistic.PoseLandmark.RIGHT_PINKY, 
                    mp_holistic.PoseLandmark.RIGHT_INDEX, 
                    ]               
                if results.pose_landmarks:
                    
                #     lndmark_list.append ([ [results.pose_landmarks.landmark[i.value].x, \
                #             results.pose_landmarks.landmark[i.value].y, \
                #             results.pose_landmarks.landmark[i.value].z] for i in mp_holistic.PoseLandmark])
                    
                    for landmark in custom_lm_styles:
                        if landmark not in included_landmarks:
                            # we change the way the excluded landmarks are drawn
                            custom_lm_styles[landmark] = DrawingSpec(color=(255,255,0), thickness=None) 
                            # we remove all connections which contain these landmarks
                            custom_connections = remove_landmark_connections(custom_connections, landmark)

                    mp_drawing.draw_landmarks(img, results.pose_landmarks, custom_connections,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                ) 
                    output_video.write(img)
                # if results.right_hand_landmarks:
                #     mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                #                 mp_drawing_styles.get_default_hand_landmarks_style(),
                #                 mp_drawing_styles.get_default_hand_connections_style()
                #                 ) 
                #     output_video.write(img)
                #     lndmark_list.append ([[results.right_hand_landmarks.landmark[i].x, \
                #                         results.right_hand_landmarks.landmark[i].y, \
                #                         results.right_hand_landmarks.landmark[i].z] for i in mp_holistic.HandLandmark])
        
                    
                else:
                    proceed_frame+=1
                    continue
                    # output_video.write(img2)
                #     lndmark_list.append(np.zeros((21, 3)))
        # np_path = output_path + ".npy"
        # np.save(np_path, lndmark_list)  
        # print((lndmark_list[0]))
        # cap.release()
        end_time = time.time()
        print(proceed_frame)
        execution_time = end_time - start_time
        print("程式執行時間：", execution_time, "秒")
        # output_video.release()
        # cv2.destroyAllWindows()


detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/bhc_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/bhc_left_1')
detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/bhc_left_2_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/bhc_left_2')
detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/bhpull_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/bhpull_left_1')
detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/bhpush_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/bhpush_left_1')
# detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/bht_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/bht_left_1')
# detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/bht_left_2_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/bht_left_2')
# detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/fhc_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/fhc_left_1')
# detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/fhpull_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/fhpull_left_1')
# detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/fhpush_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/fhpush_left_1')
# detect_right_hand(r'/home/chenzy/FastInst-main/new_record/left_player/right/fhs_left_1_crop.mp4',r'/home/chenzy/FastInst-main/mediapipe/new_record/left/right_handed/arm_palm/fhs_left_1')

