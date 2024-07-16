from moviepy.editor import *
   
# loading video gfg 
clip_1 = VideoFileClip(r"D:\thesis\video\left\2_record\00011.MTS") 
clip_2 = VideoFileClip(r"D:\thesis\video\left\2_record\00012.MTS") 
clip_3 = VideoFileClip(r"D:\thesis\video\left\2_record\00013.MTS") 
# getting only first 5 seconds and save video
# clip = clip.subclip(0,334 ) 
# path = "D:\thesis\video\trim\left\right_handed"
def cut_video_and_save(clip, start, end, dict) :
    clip = clip.subclip(start, end)
    clip.write_videofile(dict)
cut_video_and_save(clip_1, 37, 49, r"D:\thesis\video\trim\left\right_handed\bht_left_1.mp4")
cut_video_and_save(clip_1, 64, 67, r"D:\thesis\video\trim\left\right_handed\bht_left_2.mp4")
cut_video_and_save(clip_2, 60, 65, r"D:\thesis\video\trim\left\right_handed\bhc_left_1.mp4")
cut_video_and_save(clip_2, 80, 87, r"D:\thesis\video\trim\left\right_handed\bhc_left_2.mp4")
cut_video_and_save(clip_2, 94, 100, r"D:\thesis\video\trim\left\right_handed\fhc_left_1.mp4")
cut_video_and_save(clip_3, 188, 193, r"D:\thesis\video\trim\left\right_handed\bhpush_left_1.mp4")
cut_video_and_save(clip_3, 212, 219, r"D:\thesis\video\trim\left\right_handed\fhpush_left_1.mp4")
cut_video_and_save(clip_3, 230, 238, r"D:\thesis\video\trim\left\right_handed\bhpull_left_1.mp4")
cut_video_and_save(clip_3, 257, 267, r"D:\thesis\video\trim\left\right_handed\fhpull_left_1.mp4")
cut_video_and_save(clip_3, 300, 310, r"D:\thesis\video\trim\left\right_handed\fhs_left_1.mp4")
# cut_video_and_save(clip_3, 115, 121, r"D:\thesis\video\trim\left\right_handed\fhpull_left_2.mp4")
# cut_video_and_save(clip_3, 137, 148, r"D:\thesis\video\trim\left\right_handed\fhs_left_1.mp4")
# cut_video_and_save(clip_3, 150, 155, r"D:\thesis\video\trim\left\right_handed\fhs_left_2.mp4")