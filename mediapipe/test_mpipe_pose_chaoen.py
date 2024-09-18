import csv
import json
import time

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe import solutions as mp_pose

# 初始化 Mediapipe Pose 模組
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 定義每個 PoseLandmark 成員對應的器官名稱
landmark_names = {
    mp_pose.PoseLandmark.NOSE: "Nose",
    mp_pose.PoseLandmark.LEFT_EYE_INNER: "Left Eye Inner",
    mp_pose.PoseLandmark.LEFT_EYE: "Left Eye",
    mp_pose.PoseLandmark.LEFT_EYE_OUTER: "Left Eye Outer",
    mp_pose.PoseLandmark.RIGHT_EYE_INNER: "Right Eye Inner",
    mp_pose.PoseLandmark.RIGHT_EYE: "Right Eye",
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER: "Right Eye Outer",
    mp_pose.PoseLandmark.LEFT_EAR: "Left Ear",
    mp_pose.PoseLandmark.RIGHT_EAR: "Right Ear",
    mp_pose.PoseLandmark.LEFT_SHOULDER: "Left Shoulder",
    mp_pose.PoseLandmark.RIGHT_SHOULDER: "Right Shoulder",
    mp_pose.PoseLandmark.LEFT_ELBOW: "Left Elbow",
    mp_pose.PoseLandmark.RIGHT_ELBOW: "Right Elbow",
    mp_pose.PoseLandmark.LEFT_WRIST: "Left Wrist",
    mp_pose.PoseLandmark.RIGHT_WRIST: "Right Wrist",
    mp_pose.PoseLandmark.LEFT_PINKY: "Left Pinky",
    mp_pose.PoseLandmark.RIGHT_PINKY: "Right Pinky",
    mp_pose.PoseLandmark.LEFT_INDEX: "Left Index",
    mp_pose.PoseLandmark.RIGHT_INDEX: "Right Index",
    mp_pose.PoseLandmark.LEFT_THUMB: "Left Thumb",
    mp_pose.PoseLandmark.RIGHT_THUMB: "Right Thumb",
    mp_pose.PoseLandmark.LEFT_HIP: "Left Hip",
    mp_pose.PoseLandmark.RIGHT_HIP: "Right Hip",
    mp_pose.PoseLandmark.LEFT_KNEE: "Left Knee",
    mp_pose.PoseLandmark.RIGHT_KNEE: "Right Knee",
    mp_pose.PoseLandmark.LEFT_ANKLE: "Left Ankle",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "Right Ankle",
    mp_pose.PoseLandmark.LEFT_HEEL: "Left Heel",
    mp_pose.PoseLandmark.RIGHT_HEEL: "Right Heel",
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX: "Left Foot Index",
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: "Right Foot Index",
}


def draw_skeleton_on_black(image_shape, detection_result):
    skeleton_image = np.zeros(image_shape, dtype=np.uint8)  # Create a black background
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(
        skeleton_image,
        detection_result,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
    )
    return skeleton_image


def detect_pose(video_path, paddle_video_path, output_video_path, npy_path=None):
    proceed_frame = 0
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    paddle_cap = cv2.VideoCapture(paddle_video_path)
    total_frames = int(paddle_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    str_path = video_path.split("/")

    file_name = str_path[-1].replace(".mp4", "")
    output_video_path = output_video_path + file_name + ".mp4"
    # npy_path = npy_path + file_name+ ".npy"

    fps, width, height = (
        paddle_cap.get(cv2.CAP_PROP_FPS),
        int(paddle_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(paddle_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video_path, codec, fps, (int(width), height), isColor=True)
    print(fps, width, height)
    image_shape = (height, width, 3)  # Get the shape of the input image
    lndmark_list = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        for _ in tqdm(range(total_frames), desc="Processing Frames"):
            ret, img = cap.read()
            if not ret:
                print(f"Cannot receive {video_path} frame")
                break
            ret, paddle_img = paddle_cap.read()
            if not ret:
                print(f"Cannot receive {paddle_video_path} frame")
                break

            else:
                proceed_frame += 1
                # 分割影像的左半部和右半部
                height, width, _ = img.shape
                left_half = img[:, : width // 2]  # 取得左半部影像
                right_half = img[:, width // 2 :]  # 取得右半部影像
                paddle_left_half = paddle_img[:, : width // 2]  # 取得左半部影像
                paddle_right_half = paddle_img[:, width // 2 :]  # 取得右半部影像

                # 對左半部進行姿勢偵測
                results = pose.process(left_half)  # 取得姿勢偵測結果
                # skeleton_image = draw_skeleton_on_black(image_shape, results.pose_landmarks)

                # 根據姿勢偵測結果，標記身體節點和骨架
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        paddle_left_half,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    landmarks_dict = {
                        landmark_names.get(mp_pose.PoseLandmark(i), f"Landmark_{i}"): {
                            "x": results.pose_landmarks.landmark[i.value].x,
                            "y": results.pose_landmarks.landmark[i.value].y,
                            "z": results.pose_landmarks.landmark[i.value].z,
                        }
                        for i in mp_pose.PoseLandmark
                    }
                    lndmark_list.append({"Frame": proceed_frame, "Landmarks": landmarks_dict})
                else:
                    landmarks_dict = {
                        landmark_names.get(mp_pose.PoseLandmark(i), f"Landmark_{i}"): {"x": -1.0, "y": -1.0, "z": -1.0}
                        for i in mp_pose.PoseLandmark
                    }
                    lndmark_list.append({"Frame": proceed_frame, "Landmarks": landmarks_dict})

                # 將左半部和右半部重新合併
                paddle_img[:, : width // 2] = paddle_left_half  # 用處理過的左半部替換原來的左半部

                # 將合併的影像寫入影片
                output_video.write(paddle_img)

        path_csv_str = output_video_path + "pose_data.json"
        with open(path_csv_str, "w") as jsonfile:
            json.dump(lndmark_list, jsonfile, indent=4)

        cap.release()
        output_video.release()
    end_time = time.time()
    print(proceed_frame)
    execution_time = end_time - start_time
    print("程式執行時間：", execution_time, "秒")


detect_pose(
    "test_videos/video.MOV",
    "test_videos/video_2024_09_13_17_27_33_-13-Sep-2024-17f-27.mp4",
    "mediapipe/ideathon/",
)

# detect_pose("new_record/left_player/right/bhc_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/bhc_left_2_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/bhpull_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/bhpush_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/bht_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/bht_left_2_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/fhc_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/fhpull_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/fhpush_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left_player/right/fhs_left_1_crop.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose/")
# detect_pose("new_record/left/fhpush_left_1.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose")
# detect_pose("new_record/left/fhs_left_1.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose")
# detect_pose("new_record/left/fhs_left_2.mp4", "mediapipe/new_record/left/right_handed/test_pose/", "mediapipe/new_record/left/right_handed/pose")
