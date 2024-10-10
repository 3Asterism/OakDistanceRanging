#!/usr/bin/env python3
# coding=utf-8
from __future__ import annotations

import collections
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

FPS = 15

lower_threshold = 0  # mm
upper_threshold = 100_000  # mm

num_classes = 80

blob = Path(__file__).parent.joinpath("yolov8n_openvino_2021.4_6shave.blob")
model = dai.OpenVINO.Blob(blob)
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

output_name, output_tenser = next(iter(model.networkOutputs.items()))
num_classes = output_tenser.dims[2] - 5 if "yolov6" in output_name else output_tenser.dims[2] // 3 - 5

# fmt: off
label_map = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
# fmt: on


calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
top_left = dai.Point2f(0.4, 0.4)
bottom_right = dai.Point2f(0.6, 0.6)
config = dai.SpatialLocationCalculatorConfigData()


class FPSHandler:
    """
    Class that handles all FPS-related operations. Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on it's FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fpsBgColor = (0, 0, 0)
    _fpsColor = (255, 255, 255)
    _fpsType = cv2.FONT_HERSHEY_SIMPLEX
    _fpsLineType = cv2.LINE_AA

    def __init__(self, cap=None, maxTicks=100):
        """
        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            maxTicks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if maxTicks < 2:
            msg = f"Proviced maxTicks value must be 2 or higher (supplied: {maxTicks})"
            raise ValueError(msg)

        self._maxTicks = maxTicks

    def nextIter(self):
        """
        Marks the next iteration of the processing loop. Will use :obj:`time.sleep` method if initialized with video file
        object
        """
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frameDelay = 1.0 / self._framerate
            delay = (self._timestamp + frameDelay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tickFps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            timeDiff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / timeDiff if timeDiff != 0 else 0.0
        return 0.0

    def fps(self):
        """
        Calculates FPS value based on :func:`nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        timeDiff = self._timestamp - self._start
        return self._iterCnt / timeDiff if timeDiff != 0 else 0.0

    def printStatus(self):
        """Prints total FPS for all names stored in :func:`tick` calls"""
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tickFps(name):.1f}")

    def drawFps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frameFps = f"{name.upper()} FPS: {round(self.tickFps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, frameFps, (5, 15), self._fpsType, 0.5, self._fpsBgColor, 4, self._fpsLineType)
        cv2.putText(frame, frameFps, (5, 15), self._fpsType, 0.5, self._fpsColor, 1, self._fpsLineType)

        if "nn" in self._ticks:
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tickFps('nn'), 1)}",
                (5, 30),
                self._fpsType,
                0.5,
                self._fpsBgColor,
                4,
                self._fpsLineType,
            )
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tickFps('nn'), 1)}",
                (5, 30),
                self._fpsType,
                0.5,
                self._fpsColor,
                1,
                self._fpsLineType,
            )


def create_pipeline():
    global calculation_algorithm, config

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    left = pipeline.create(dai.node.ColorCamera)

    tof = pipeline.create(dai.node.ToF)
    camTof = pipeline.create(dai.node.Camera)
    imageAlign = pipeline.create(dai.node.ImageAlign)

    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

    imageOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)

    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

    imageOut.setStreamName("image")
    disparityOut.setStreamName("disp")
    xoutNN.setStreamName("detections")

    xoutSpatialData.setStreamName("spatial_data")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # Properties
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    left.setFps(FPS)
    left.setPreviewSize(W, H)
    left.setPreviewKeepAspectRatio(False)
    left.setInterleaved(False)
    left.setIspScale(1, 2)

    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    # ToF settings
    camTof.setFps(FPS)
    camTof.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    camTof.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    tof.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)

    # Image align
    imageAlign.setOutputSize(640, 400)
    left.setVideoSize(640, 400)

    # Network specific settings
    spatialDetectionNetwork.setBlob(model)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(num_classes)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([])
    spatialDetectionNetwork.setAnchorMasks({})
    spatialDetectionNetwork.setIouThreshold(0.3)

    # spatial specific parameters
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(lower_threshold)
    spatialDetectionNetwork.setDepthUpperThreshold(upper_threshold)
    spatialDetectionNetwork.setSpatialCalculationAlgorithm(calculation_algorithm)

    # Config
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = lower_threshold
    config.depthThresholds.upperThreshold = upper_threshold
    config.calculationAlgorithm = calculation_algorithm
    config.roi = dai.Rect(top_left, bottom_right)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    # Linking
    camTof.raw.link(tof.input)
    tof.depth.link(imageAlign.input)
    left.video.link(imageOut.input)
    left.isp.link(imageAlign.inputAlignTo)
    left.preview.link(spatialDetectionNetwork.input)

    spatialDetectionNetwork.passthroughDepth.link(disparityOut.input)
    imageAlign.outputAligned.link(spatialDetectionNetwork.inputDepth)


    spatialDetectionNetwork.passthroughDepth.link(spatialLocationCalculator.inputDepth)
    spatialDetectionNetwork.out.link(xoutNN.input)

    spatialLocationCalculator.out.link(xoutSpatialData.input)

    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    return pipeline


def check_input(roi, frame, DELTA=5):
    """Check if input is ROI or point. If point, convert to ROI"""
    # Convert to a numpy array if input is a list
    if isinstance(roi, list):
        roi = np.array(roi)

    # Limit the point so ROI won't be outside the frame
    if roi.shape in {(2,), (2, 1)}:
        roi = np.hstack([roi, np.array([[-DELTA, -DELTA], [DELTA, DELTA]])])
    elif roi.shape in {(4,), (4, 1)}:
        roi = np.array(roi)

    roi.clip([DELTA, DELTA], [frame.shape[1] - DELTA, frame.shape[0] - DELTA])

    return roi / frame.shape[1::-1]


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global ref_pt, click_roi
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt = [(x, y)]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_pt.append((x, y))
        ref_pt = np.array(ref_pt)
        click_roi = np.array([np.min(ref_pt, axis=0), np.max(ref_pt, axis=0)])


def run():
    global ref_pt, click_roi, calculation_algorithm, config
    # Connect to device and start pipeline
    with dai.Device(dai.DeviceInfo("169.254.1.222")) as device:
        pipeline = create_pipeline()
        device.startPipeline(pipeline)

        frame_rgb = None
        depth_frame = None
        depth_datas = []
        detections = []
        bbox_colors = np.random.default_rng().integers(256, size=(num_classes, 3)).tolist()
        step_size = 0.01
        new_config = False

        # Configure windows; trackbar adjusts blending ratio of rgb/depthQueueData
        rgb_window_name = "image"
        depth_window_name = "depthQueueData"
        cv2.namedWindow(rgb_window_name)
        cv2.namedWindow(depth_window_name)

        cv2.setMouseCallback(rgb_window_name, click_and_crop)
        cv2.setMouseCallback(depth_window_name, click_and_crop)

        print("Use WASD keys to move ROI!")

        spatial_calc_config_in_queue = device.getInputQueue("spatialCalcConfig")
        image_queue = device.getOutputQueue("image")
        disp_queue = device.getOutputQueue("disp")
        spatial_data_queue = device.getOutputQueue("spatial_data")
        detect_queue = device.getOutputQueue(name="detections")

        def frame_norm(frame, bbox):
            """NN 数据作为边界框位置，位于 <0..1> 范围内它们需要用帧宽度/高度进行归一化"""
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def draw_text(frame, text, org, color=(255, 255, 255), thickness=1):
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)

        def draw_rect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
            cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)
            cv2.rectangle(frame, topLeft, bottomRight, color, thickness)

        def draw_detection(frame, detections):
            for detection in detections:
                bbox = frame_norm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )
                draw_text(
                    frame,
                    label_map[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                )
                draw_text(
                    frame,
                    f"{detection.confidence:.2%}",
                    (bbox[0] + 10, bbox[1] + 35),
                )
                draw_rect(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_colors[detection.label], 1)
                if hasattr(detection, "spatialCoordinates"):
                    draw_text(
                        frame,
                        f"X: {int(detection.spatialCoordinates.x)} mm",
                        (bbox[0] + 10, bbox[1] + 50),
                    )
                    draw_text(
                        frame,
                        f"Y: {int(detection.spatialCoordinates.y)} mm",
                        (bbox[0] + 10, bbox[1] + 65),
                    )
                    draw_text(
                        frame,
                        f"Z: {int(detection.spatialCoordinates.z)} mm",
                        (bbox[0] + 10, bbox[1] + 80),
                    )

        def draw_spatial_locations(frame, spatialLocations):
            for depthData in spatialLocations:
                roi = depthData.config.roi
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 4)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                draw_text(
                    frame,
                    f"X: {int(depthData.spatialCoordinates.x)} mm",
                    (xmin + 10, ymin + 20),
                )
                draw_text(
                    frame,
                    f"Y: {int(depthData.spatialCoordinates.y)} mm",
                    (xmin + 10, ymin + 35),
                )
                draw_text(
                    frame,
                    f"Z: {int(depthData.spatialCoordinates.z)} mm",
                    (xmin + 10, ymin + 50),
                )

        fps = FPSHandler()

        disp_data = None

        while not device.isClosed():
            image_data = image_queue.tryGet()
            disp_data = disp_queue.tryGet()
            spatial_data = spatial_data_queue.tryGet()
            det_data = detect_queue.tryGet()

            if spatial_data is not None:
                depth_datas = spatial_data.getSpatialLocations()

            if det_data is not None:
                fps.tick("nn")
                detections = det_data.detections

            if image_data is not None:
                frame_rgb = image_data.getCvFrame()
                draw_detection(frame_rgb, detections)
                draw_spatial_locations(frame_rgb, depth_datas)
                fps.tick("image")
                fps.drawFps(frame_rgb, "image")

                cv2.imshow(rgb_window_name, frame_rgb)

            if disp_data is not None:
                depth_frame = disp_data.getFrame()

                depth_downscaled = depth_frame[::4]
                if np.all(depth_downscaled == 0):
                    min_depth = 0  # Set a default minimum depth value when all elements are zero
                else:
                    min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                max_depth = np.percentile(depth_downscaled, 99)
                depth_frame_color = np.interp(depth_frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

                depth_frame = np.ascontiguousarray(depth_frame_color)
                draw_detection(depth_frame, detections)
                draw_spatial_locations(depth_frame, depth_datas)
                fps.tick("dispData")
                fps.drawFps(depth_frame, "dispData")
                cv2.imshow(depth_window_name, depth_frame)

            # Blend when both received
            if frame_rgb is not None and depth_frame is not None and click_roi is not None:
                (
                    [top_left.x, top_left.y],
                    [
                        bottom_right.x,
                        bottom_right.y,
                    ],
                ) = check_input(click_roi, frame_rgb)
                click_roi = None
                new_config = True

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("w"):
                if top_left.y - step_size >= 0:
                    top_left.y -= step_size
                    bottom_right.y -= step_size
                    new_config = True
            elif key == ord("a"):
                if top_left.x - step_size >= 0:
                    top_left.x -= step_size
                    bottom_right.x -= step_size
                    new_config = True
            elif key == ord("s"):
                if bottom_right.y + step_size <= 1:
                    top_left.y += step_size
                    bottom_right.y += step_size
                    new_config = True
            elif key == ord("d"):
                if bottom_right.x + step_size <= 1:
                    top_left.x += step_size
                    bottom_right.x += step_size
                    new_config = True

            elif key == ord("1"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
                print("Switching calculation algorithm to MEAN!")
                new_config = True
            elif key == ord("2"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
                print("Switching calculation algorithm to MIN!")
                new_config = True
            elif key == ord("3"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MAX
                print("Switching calculation algorithm to MAX!")
                new_config = True
            elif key == ord("4"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
                print("Switching calculation algorithm to MODE!")
                new_config = True
            elif key == ord("5"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                print("Switching calculation algorithm to MEDIAN!")
                new_config = True

            if new_config:
                # config.depthThresholds.lowerThreshold = lowerThreshold
                # config.depthThresholds.upperThreshold = upperThreshold
                config.roi = dai.Rect(top_left, bottom_right)
                config.calculationAlgorithm = calculation_algorithm
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatial_calc_config_in_queue.send(cfg)
                new_config = False


if __name__ == "__main__":
    ref_pt = None
    click_roi = None
    run()
