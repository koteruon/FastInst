from moviepy.editor import *
import cv2
import numpy as np

def extract_frame(clip_path, output_path):
    clip = cv2.VideoCapture(clip_path)
    success,image = clip.read()
    count = 0
    while success:
        if count%15 ==0 :
            cv2.imwrite(output_path + "_frame%d.jpg" % count, image)     # save frame as JPEG file     
            success,image = clip.read()
            # print('Read a new frame: ', success)
            count += 1
        else:
            success,image = clip.read()
            count +=1
    print("done")

# extract_frame(r"D:\thesis\video\trim\left\right_handed\bht_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop_extract\bht_left_1")
# extract_frame(r"D:\thesis\video\trim\left\right_handed\bht_left_2.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bht_left_2")
extract_frame( r"D:\thesis\video\trim\left\right_handed_crop\bhc_left_1_crop.mp4", r"D:\thesis\video\trim\left\right_handed_crop_extract\bhc_left_1")
# extract_frame( r"D:\thesis\video\trim\left\right_handed\bhc_left_2.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhc_left_2")
# extract_frame( r"D:\thesis\video\trim\left\right_handed\fhc_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop_extract\fhc_left_1")
# extract_frame( r"D:\thesis\video\trim\left\right_handed\bhpush_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhpush_left_1")
# extract_frame(r"D:\thesis\video\trim\left\right_handed\fhpush_left_1.mp4", r"D:\thesis\video\trim\left\right_handed_crop_extract\fhpush_left_1")
extract_frame(r"D:\thesis\video\trim\left\right_handed_crop\bhpull_left_1_crop.mp4", r"D:\thesis\video\trim\left\right_handed_crop\bhpull_left_1")
extract_frame(r"D:\thesis\video\trim\left\right_handed_crop\fhpull_left_1_crop.mp4", r"D:\thesis\video\trim\left\right_handed_crop_extract\fhpull_left_1")
extract_frame(r"D:\thesis\video\trim\left\right_handed_crop\fhs_left_1_crop.mp4", r"D:\thesis\video\trim\left\right_handed_crop_extract\fhs_left_1")

# cut_video_and_save(clip_3, 115, 121, r"D:\thesis\video\trim\left\right_handed\fhpull_left_2.mp4")
# cut_video_and_save(clip_3, 137, 148, r"D:\thesis\video\trim\left\right_handed\fhs_left_1.mp4")
# cut_video_and_save(clip_3, 150, 155, r"D:\thesis\video\trim\left\right_handed\fhs_left_2.mp4")