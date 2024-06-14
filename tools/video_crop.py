from moviepy.editor import *
import cv2
import numpy as np

# loading video gfg 
# clip = cv2.VideoCapture(r'D:\thesis\video\trim\left\right_handed\bhc_left_1.mp4')
# getting only first 5 seconds and save video
# clip = clip.subclip(0,334 ) 
# path = "D:\thesis\video\trim\left\right_handed"
def crop_video(clip, output_path):
    video = cv2.VideoCapture(clip)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = output_path
    
    output_file = cv2.VideoWriter(
                filename=f"{basename}_crop.mp4",
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                # fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                # frameSize=(width, height),
                frameSize=(1600, 1080 ),
                isColor=True,
            )
    try:
        while True:
            ret, frame = video.read()
            # (height, width) = frame.shape[:2]
            sky = frame[0:1080, 0:1600]
            # cv2.imshow('Video', sky)
            output_file.write(sky)
    except:
        print("output")
    # output_file.release()

crop_video(r"D:\thesis\video\trim\left\right_handed\bht_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bht_left_1")
crop_video(r"D:\thesis\video\trim\left\right_handed\bht_left_2.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bht_left_2")
crop_video( r"D:\thesis\video\trim\left\right_handed\bhc_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhc_left_1")
crop_video( r"D:\thesis\video\trim\left\right_handed\bhc_left_2.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhc_left_2")
crop_video( r"D:\thesis\video\trim\left\right_handed\fhc_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\fhc_left_1")
crop_video( r"D:\thesis\video\trim\left\right_handed\bhpush_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhpush_left_1")
crop_video(r"D:\thesis\video\trim\left\right_handed\fhpush_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\fhpush_left_1")
crop_video(r"D:\thesis\video\trim\left\right_handed\bhpull_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhpull_left_1")
crop_video(r"D:\thesis\video\trim\left\right_handed\fhpull_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\fhpull_left_1")
crop_video(r"D:\thesis\video\trim\left\right_handed\fhs_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\fhs_left_1")

# cut_video_and_save(clip_3, 115, 121, r"D:\thesis\video\trim\left\right_handed\fhpull_left_2.mp4")
# cut_video_and_save(clip_3, 137, 148, r"D:\thesis\video\trim\left\right_handed\fhs_left_1.mp4")
# cut_video_and_save(clip_3, 150, 155, r"D:\thesis\video\trim\left\right_handed\fhs_left_2.mp4")