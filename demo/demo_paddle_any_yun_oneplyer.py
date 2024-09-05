# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import csv
import json
import tempfile
import warnings
from datetime import datetime

import cv2
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from tqdm import tqdm

from fastinst import add_fastinst_config

# constants
WINDOW_NAME = "Table Tennis detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_fastinst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; " "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--csv_name",
        default="",
        help="csv_name",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, detect_left=False)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            #             img = read_image(path, format="BGR")
            # OneNet uses RGB input as default
            img = read_image(path, format="RGB")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    (
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished"
                    ),
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.output) > 0, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm(demo.run_on_video(cam, args.confidence_threshold)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        codec, file_ext = ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")

        now = datetime.now()  # current date and time
        time_name = now.strftime("_%Y_%m_%d_%H_%M_%S")
        time_str = str(time_name)

        # merge_mask video
        # fourcc_merge = cv2.VideoWriter_fourcc(*'mp4v')
        # frames_per_second_merge = video.get(cv2.CAP_PROP_FPS)
        # out_merge = cv2.VideoWriter('/home/chenzy/FastInst-main/output/' + time_str+ basename + '.mp4', fourcc_merge, frames_per_second_merge, (640, 640), False)
        # # False: grayscale      # out_merge = cv2.VideoWriter('./table-tennis/label_mask/maskmerge_' + basename + '.mp4', fourcc_merge, frames_per_second_merge, (640, 640))

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + time_str + file_ext
                print(output_fname)
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname

            dateTimeObj = datetime.now()
            output_file = cv2.VideoWriter(
                filename=f"{output_fname.split('.')[-2]}_{dateTimeObj.strftime('-%d-%b-%Y-%Hf-%M')}.mp4",
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                # fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)

        paddle_L_area_list = []

        bb_boxes_pad_L_center_list = []

        # Create an empty DataFrame
        csv_data = pd.DataFrame(columns=["Frame", "Paddle_L_Pixel"])

        durations = []
        frame_count = 0
        for vis_frame, pred in tqdm(demo.run_on_video(video, args.confidence_threshold, 1), total=num_frames):
            # Calculate frames per second
            frame_count += 1

            # 1. demo
            # 2. d2_predictor.run_on_video -> selfpredictor.get() -> predictions -> def process_predictions
            # 3. video_visilizer.draw_instance_predictions -> vis_frame, pred

            # Calculate only paddle(e.g: class=3)
            pred_c = pred.pred_classes.numpy()
            pred_m = pred.pred_masks.numpy()

            # id 0 paddle
            # Calculate paddle location (class=3), person location (class=1)
            pred_m_class = pred_c.copy()
            pred_m_pad_arr = pred_m.copy()
            pred_m_pad_arr = pred_m_pad_arr[np.where(pred_m_class == 0)]  # paddle

            paddle_L_mask = []
            paddle_L_pixel = []

            bb_boxs_pad = []
            bb_boxs_person = []

            max_paddle_box = -1
            bb_boxes_pad = None
            for pad_idx, pred_m_pad in enumerate(pred_m_pad_arr):
                label_mask_pad = pred_m_pad.astype(np.uint8)
                contours, hierarchy = cv2.findContours(label_mask_pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                bb_boxes_pads = [cv2.boundingRect(cnt) for cnt in contours]
                max_idx = 0
                if len(bb_boxes_pads) == 0:
                    print("bbox_pad None")
                    continue
                elif len(bb_boxes_pads) >= 2:
                    max_box = 0
                    for box_idx, bb_box in enumerate(bb_boxes_pads):
                        if bb_box[2] * bb_box[3] > max_box:
                            max_box = bb_box[2] * bb_box[3]
                            max_idx = box_idx
                else:
                    max_box = bb_boxes_pads[max_idx][2] * bb_boxes_pads[max_idx][3]

                # find the maximum paddle box
                if max_paddle_box < max_box:
                    max_paddle_box = max_box
                    bb_boxes_pad = bb_boxes_pads[max_idx]

            # only one paddle
            if bb_boxes_pad is not None and 0 <= bb_boxes_pad[0] <= width:
                label_mask_pad[label_mask_pad == 1] = 255  # imgshow grayscale
                r = label_mask_pad.copy()
                g = label_mask_pad.copy()
                b = label_mask_pad.copy()
                r[r == 1] = 255
                label_mask_pad_img = np.stack((b, g, r), axis=2)

                paddle_L_pixel.append(pred_m_pad_arr[pad_idx])
                paddle_L_mask.append(
                    label_mask_pad
                )  # paddle_L_mask.append(cropped_image_pad)  paddle_L_mask.append(label_mask_pad_img)
                bb_boxs_pad.append(bb_boxes_pad)

                # bb_boxes_pad_L_center
                bb_pad_center_x = bb_boxes_pad[0] + bb_boxes_pad[2] / 2  # x
                bb_pad_center_y = bb_boxes_pad[1] + bb_boxes_pad[3] / 2  # y

                bb_boxes_pad_L_center_list.append((bb_pad_center_x, bb_pad_center_y))
            else:
                bb_boxes_pad_L_center_list.append((bb_pad_center_x, bb_pad_center_y))

            # Write every frame of R_Paddle & R_Person Merge mask
            # Calculate the size of each merge_mask bbox
            bb_boxs_person.extend(bb_boxs_pad)
            if len(bb_boxs_person) >= 1:
                min_box_x = bb_boxs_person[0][0]
                min_box_y = bb_boxs_person[0][1]
                max_width = bb_boxs_person[0][2]
                max_height = bb_boxs_person[0][3]
                for box_idx, bb_box in enumerate(bb_boxs_person):
                    if bb_box[0] < min_box_x:
                        min_box_x = bb_box[0]
                        # print('min_box_x: ', min_box_x)                     # find the min_x (left) of merge_img
                    if bb_box[1] < min_box_y:
                        min_box_y = bb_box[1]
                        # print('mix_box_y: ', min_box_y)                     # find the min_y (top) of merge_img
                    if bb_box[2] > max_width:
                        max_width = bb_box[2]
                        # print('max_width: ', max_width)                     # find the max_width (right) of merge_img
                    if bb_box[3] > max_height:
                        max_height = bb_box[3]
                        # print('max_height: ', max_height)                   # find the max_height (bottom) of merge_img
            else:
                min_box_x, min_box_y = 0, 0
                max_width, max_height = height, width
            left = min_box_x
            top = min_box_y
            right = left + max_width
            bottom = top + max_height

            if len(paddle_L_mask) > 0:
                merge_over_lap = paddle_L_mask[0].copy()
                over_lap_img = paddle_L_mask[0].copy()
                merge_mask = paddle_L_mask[0].copy()

                for i in range(1, len(paddle_L_mask)):
                    merge_mask += paddle_L_mask[i]

                cropped_merge = merge_mask[
                    top - 50 : bottom + 50, left - 50 : right + 50
                ]  # crop merge_img bb_box range
                # Calculate the required padding
                pad_height = max(0, 640 - cropped_merge.shape[0])
                pad_width = max(0, 640 - cropped_merge.shape[1])
                # Calculate the padding amounts for top, bottom, left, and right
                top_pad = pad_height // 2
                bottom_pad = pad_height - top_pad
                left_pad = pad_width // 2
                right_pad = pad_width - left_pad
                # Pad the image
                # padded_cropped_merge = np.pad(
                #     cropped_merge, ((top_pad, bottom_pad), (left_pad, right_pad)), mode="constant"
                # )
                # (0, 0): grayscale   # padded_cropped_merge = np.pad(cropped_merge, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
                # print('padded_cropped_merge', padded_cropped_merge.shape)
                # img_frame_path = '/home/chenzy/FastInst-main/output/test_img/' + time_str
                # if os.path.isdir(img_frame_path) != True:
                #     os.makedirs(img_frame_path)
                # cv2.imwrite(img_frame_path + '/frame_' + str(frame_count) + '.png', padded_cropped_merge)
                # cv2.putText(padded_cropped_merge, "Frame: " + str(frame_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_4)
                # cv2.namedWindow('merge_mask', 0)
                # cv2.imshow('merge_mask', padded_cropped_merge)
                # cv2.imwrite('/home/chenzy/FastInst-main/output/test_img/overlap_' + time_str + '.png', blank_img)
                # out_merge.write(padded_cropped_merge)             # write label mask video
            else:
                print("Left Paddle Area is None")
                # create a black image
                black_img = np.zeros(
                    (640, 640), dtype=np.uint8
                )  # black_img = np.zeros((height, 960, 3), dtype = np.uint8)
                # cv2.imwrite('./table-tennis/label_mask/merge_R/merge_R_' + 'frame_' + str(frame_count) + '.png', black_img)
                # cv2.putText(black_img, "Frame: " + str(frame_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_4)
                # cv2.namedWindow('merge_mask', 0)
                # cv2.imshow('merge_mask', black_img)
                # out_merge.write(black_img)
            # print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

            if args.output:
                cv2.putText(
                    vis_frame,
                    "Frame: " + str(frame_count),
                    (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 0, 0),
                    5,
                    cv2.LINE_4,
                )

                # Left Paddle
                if len(paddle_L_pixel) == 0:
                    paddle_L_str = ""
                else:
                    path_str = args.output + "_left_paddle_" + time_str + ".csv"
                    paddle_L_str = str(paddle_L_pixel[0].sum())
                    with open(path_str, "w") as f_L:
                        csv_write_L = csv.writer(f_L)
                        csv_write_L.writerow(paddle_L_str)
                cv2.putText(
                    vis_frame,
                    "Left Paddle Area: " + paddle_L_str,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    5,
                    cv2.LINE_4,
                )

                # Build paddle area table in video
                if len(paddle_L_pixel) == 0:
                    paddle_L_area = 0
                else:
                    paddle_L_area = paddle_L_pixel[0].sum()

                # Append the paddle area to the list
                paddle_L_area_list.append(paddle_L_area)

                # Set the desired plot size
                plot_width = width // 2
                plot_height = height // 3
                plot_x = width // 4
                plot_y = 0

                # Determine the x-axis limits based on the frame count
                if frame_count > 120:
                    x_min = frame_count - 120
                    x_max = frame_count
                else:
                    x_min = 0
                    x_max = frame_count

                ##################################################################################################################

                # Extract the area data points for plotting
                x_data = range(x_min, x_max)
                y_data_L = paddle_L_area_list[x_min : x_max + 1]

                # Clear the previous plot
                plt.clf()
                # Initialise the subplot function using number of rows and columns
                figure, area_table = plt.subplots(1, 2, figsize=(plot_width / 100, plot_height / 100))

                # Set x-axis limits and y-axis limits for both subplots
                x_lim_area = (x_min, x_max)
                y_lim_area = (0, 3000)

                # Plot for the first subplot : Paddle_Area_L
                area_table[0].plot(x_data, y_data_L, "ro-", markersize=1)  # Paddle_Area_L : Blue
                area_table[0].set_xlim(x_lim_area)
                area_table[0].set_ylim(y_lim_area)
                area_table[0].set_xlabel("Frame")
                area_table[0].set_ylabel("Paddle Pixels Area")
                area_table[0].legend(["Left Paddle"])

                ##################################################################################################################

                # Convert the plot to an image
                fig = plt.gcf()
                fig.set_size_inches(plot_width / 100, plot_height / 100)
                fig.canvas.draw()
                plot_img = np.array(fig.canvas.renderer.buffer_rgba())
                # Resize the plot image to the desired size
                plot_img_resized = cv2.resize(plot_img, (plot_width, plot_height))
                # Convert the plot image to have the same number of color channels as the frame image
                plot_img_resized_rgb = cv2.cvtColor(plot_img_resized, cv2.COLOR_RGBA2RGB)
                # Create a copy of the frame to overlay the plot
                vis_frame_with_plot = vis_frame.copy()
                # Overlay the plot image on top of the frame
                vis_frame_with_plot[plot_y : plot_y + plot_height, plot_x : plot_x + plot_width] = plot_img_resized_rgb

                ################################################################
                # Plot paddle route

                # Determine the x-axis limits based on the frame count
                frame_count_range = 37
                if frame_count > frame_count_range:
                    frame_min_pad_L = (
                        len(bb_boxes_pad_L_center_list) - frame_count_range
                    )  # frame_min = frame_count - frame_count_range
                    frame_max = frame_count
                else:
                    frame_min_pad_L = 0
                    frame_max = frame_count

                x_data_route = range(frame_min_pad_L, frame_count)
                for i in range(frame_min_pad_L, min(frame_max, len(bb_boxes_pad_L_center_list))):
                    # Calculate the color gradient based on the frame index
                    gradient = (len(bb_boxes_pad_L_center_list) - i) / frame_count_range
                    color_component = int(255 - (gradient * 255))  # Decreasing component for gradient
                    color = (
                        0,  # b
                        max(color_component - 30, 0.0),  # g
                        color_component,  # r
                    )  # yellow to black gradient
                    center = (int(bb_boxes_pad_L_center_list[i][0]), int(bb_boxes_pad_L_center_list[i][1]))
                    cv2.circle(vis_frame_with_plot, center, 5, color, -1)

                    # Plot for the second subplot : Paddle_route
                    x_lim_route = (200, 900)  #   (250, 650)       # (450, 700)      # (300, 650)
                    y_lim_route = (200, 700)  #   (200, 500)       # (350, 600)      # (350, 650)
                    # Plot for the second subplot: Paddle_route
                    color = (
                        0,  # b
                        max((color_component - 30) / 255, 0.0),  # g
                        color_component / 255,  # r
                    )  # yellow to black gradient
                    color_normalized = color[:3]
                    for i in range(len(color_normalized)):
                        if color_normalized[i] >= 1:
                            color_normalized[i] == 1
                    area_table[1].scatter([center[0]], [height - center[1]], c=[color], marker="o", s=50)

                    area_table[1].set_xlim(x_lim_route)  # (x_lim_route) (0, 1)
                    area_table[1].set_ylim(y_lim_route)  # (y_lim_route)  (0, 1)
                    area_table[1].set_xlabel("x-position")
                    area_table[1].set_ylabel("y-position")
                    legend_L_pad = mlines.Line2D([], [], marker="o", markersize=3, label="Paddle route")  # color='cyan'
                    area_table[1].legend(handles=[legend_L_pad])

                # Convert the plot to an image
                fig = plt.gcf()
                fig.set_size_inches(plot_width / 100, plot_height / 100)
                fig.canvas.draw()
                plot_img = np.array(fig.canvas.renderer.buffer_rgba())
                # Resize the plot image to the desired size
                plot_img_resized = cv2.resize(plot_img, (plot_width, plot_height))
                # Convert the plot image to have the same number of color channels as the frame image
                plot_img_resized_rgb = cv2.cvtColor(plot_img_resized, cv2.COLOR_RGBA2RGB)
                # Create a copy of the frame to overlay the plot
                vis_frame_with_plot = vis_frame_with_plot.copy()
                # Overlay the plot image on top of the frame
                vis_frame_with_plot[plot_y : plot_y + plot_height, plot_x : plot_x + plot_width] = plot_img_resized_rgb
                output_file.write(vis_frame_with_plot)

                path_csv_str = str(args.output)
                csv_name_str = str(args.csv_name).strip()
                path_L = path_csv_str + "/" + csv_name_str + "_L_paddle.txt"

                with open(path_L, "w", newline="") as csvfile_1:
                    writer = csv.writer(csvfile_1)
                    writer.writerow(paddle_L_area_list)

                plt.close(figure)
            else:
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit

        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
