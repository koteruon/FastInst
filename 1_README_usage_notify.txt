Color code(In RGB):
Green: 46 139 87 (Nothing) | Blue: 0 154 205 (Table) | Brown: 139 115 85 (referee)
Yellow:255 215 0 (Person)  | Red:  238 0 0 (Paddle)  | Purple: 104 34 139 (scoreboard)

1: Detectron's video visulizer(In Detectron2/utills/video_visulizer) doesn't support SEGMENTATION mode with videos, hence you have to update the following code:
if self._instance_mode == ColorMode.IMAGE and self.metadata.get("thing_colors"):
            colors = [
                [x / 255 for x in self.metadata.thing_colors[c]] for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

under the line (about line 108) to make sure the color won't change during inference
labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
for more information, check https://github.com/facebookresearch/detectron2/issues/1163

also, don't forget to set the 'thing_classes' and 'thing_colors' at Detectron2/utills/visulizer 
MetadataCatalog.get("table-tennis_val").thing_classes = [' ', 'person', 'table', 'paddle', 'referee', 'scoreboard']
MetadataCatalog.get("table-tennis_val").thing_colors =[(46, 139, 87), (255, 215, 0), (0, 154, 205), (238, 0, 0), (139, 115 , 85), (104, 34, 139)]

2: There's an error in demo/predictor, don't forget to put the confidence threshold.
start at line 75, correct it with the following code.
def run_on_video(self, video, confidence_threshold):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )

            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                predictions = predictions[predictions.scores >
                                          confidence_threshold]
                vis_frame = video_visualizer.draw_instance_predictions(
                    frame, predictions)

3: To run demo_yoson, you have to update the demo/predictor function run_on_video:
   at the bottom of the function, make 
   return frame_visualizer.output
   into 
   return frame_visualizer.output, predictions
   and also updata detectron2/
   return frame_visualizer.output, predictions

4: You might encounter some weird problem, try remove tqdm function when you were running a demo code
   also, the opencv version might be too new to run, try downgrade the opencv version to 4.1 version
   or try to install opencv-contrib to solve the problem
   for more info check the website: https://blog.csdn.net/baobei0112/article/details/121648262 