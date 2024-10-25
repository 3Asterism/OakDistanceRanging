#!/usr/bin/env python3
# coding=utf-8
from __future__ import annotations

import collections
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import math

lower_threshold = 0  # 最小深度阈值，单位为毫米
upper_threshold = 100_000  # 最大深度阈值，单位为毫米

num_classes = 1  # 类别数量

# 加载模型文件
blob = Path(__file__).parent.parent.joinpath("detect/model/best_openvino_2022.1_5shave.blob")
model = dai.OpenVINO.Blob(blob)  # 创建OpenVINO模型对象
dim = next(iter(model.networkInputs.values())).dims  # 获取输入维度
W, H = dim[:2]  # 获取宽度和高度

# 获取输出名称和张量
output_name, output_tenser = next(iter(model.networkOutputs.items()))
# 根据模型类型确定类别数量
num_classes = output_tenser.dims[2] - 5 if "yolov6" in output_name else output_tenser.dims[2] // 3 - 5

# 标签映射
# fmt: off
label_map = [
    "head"
]
# fmt: on

# 最小深度设置，视差范围加倍（从95到190）：
extended_disparity = True
# 更好地处理长距离的精度，分数视差32级：
subpixel = True
# 更好地处理遮挡：
lr_check = True

# 深度计算算法设置为平均
calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
top_left = dai.Point2f(0.4, 0.4)  # 定义矩形计算区域的左上角
bottom_right = dai.Point2f(0.6, 0.6)  # 定义矩形计算区域的右下角
config = dai.SpatialLocationCalculatorConfigData()  # 创建空间位置计算配置对象


class FPSHandler:
    """
    处理所有FPS相关操作的类。主要用于计算不同流的FPS，
    也可以根据视频文件的FPS属性供给视频文件，而不是应用性能（这可以防止视频过快播放，
    如果我们提前处理完一帧而下一个视频帧应该被消费）。
    """

    _fpsBgColor = (0, 0, 0)  # FPS背景颜色
    _fpsColor = (255, 255, 255)  # FPS文本颜色
    _fpsType = cv2.FONT_HERSHEY_SIMPLEX  # FPS字体类型
    _fpsLineType = cv2.LINE_AA  # FPS线型

    def __init__(self, cap=None, maxTicks=100):
        """
        Args:
            cap (cv2.VideoCapture, Optional): 视频文件对象的处理器
            maxTicks (int, Optional): FPS计算的最大刻度数
        """
        self._timestamp = None  # 当前时间戳
        self._start = None  # 开始时间
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None  # 获取视频帧率
        self._useCamera = cap is None  # 判断是否使用摄像头

        self._iterCnt = 0  # 迭代计数
        self._ticks = {}  # 存储时间戳的字典

        if maxTicks < 2:
            msg = f"提供的maxTicks值必须大于或等于2（提供的值：{maxTicks}）"
            raise ValueError(msg)

        self._maxTicks = maxTicks  # 最大刻度数

    def nextIter(self):
        """
        标记处理循环的下一个迭代。如果初始化时使用了视频文件对象，将使用：obj:`time.sleep`方法。
        """
        if self._start is None:
            self._start = time.monotonic()  # 初始化开始时间

        if not self._useCamera and self._timestamp is not None:
            frameDelay = 1.0 / self._framerate  # 计算帧延迟
            delay = (self._timestamp + frameDelay) - time.monotonic()  # 计算需要延迟的时间
            if delay > 0:
                time.sleep(delay)  # 进行延迟
        self._timestamp = time.monotonic()  # 更新当前时间戳
        self._iterCnt += 1  # 迭代计数加一

    def tick(self, name):
        """
        为指定名称标记一个时间点。

        Args:
            name (str): 指定的时间戳名称
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)  # 初始化时间戳队列
        self._ticks[name].append(time.monotonic())  # 添加当前时间戳

    def tickFps(self, name):
        """
        根据指定名称计算FPS。

        Args:
            name (str): 指定的时间戳名称

        Returns:
            float: 计算得到的FPS，如果失败则返回：code:`0.0`
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            timeDiff = self._ticks[name][-1] - self._ticks[name][0]  # 计算时间差
            return (len(self._ticks[name]) - 1) / timeDiff if timeDiff != 0 else 0.0  # 计算FPS
        return 0.0

    def fps(self):
        """
        根据：func:`nextIter`调用计算FPS值，即处理循环的FPS。

        Returns:
            float: 计算得到的FPS，如果失败则返回：code:`0.0`
        """
        if self._start is None or self._timestamp is None:
            return 0.0  # 如果未初始化，则返回0
        timeDiff = self._timestamp - self._start  # 计算处理时间差
        return self._iterCnt / timeDiff if timeDiff != 0 else 0.0  # 返回FPS

    def printStatus(self):
        """打印在：func:`tick`调用中存储的所有名称的总FPS"""
        print("=== 总FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tickFps(name):.1f}")  # 打印每个名称的FPS

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

        if "nn" in self._ticks:  # 检查“nn”时间戳是否存在
            # 在帧上绘制神经网络的FPS信息
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tickFps('nn'), 1)}",  # 计算并格式化FPS
                (5, 30),  # 文本位置（左上角）
                self._fpsType,  # 字体类型
                0.5,  # 字体大小
                self._fpsBgColor,  # 背景颜色
                4,  # 字体厚度
                self._fpsLineType,  # 线型
            )
            # 重新绘制FPS文本，以便在前景中显示
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tickFps('nn'), 1)}",
                (5, 30),
                self._fpsType,
                0.5,
                self._fpsColor,  # 前景颜色
                1,  # 字体厚度
                self._fpsLineType,
            )


def create_pipeline():
    global calculation_algorithm, config  # 声明全局变量

    # 创建管道
    pipeline = dai.Pipeline()

    # 定义源节点和输出节点
    left = pipeline.create(dai.node.ColorCamera)  # 创建左摄像头
    right = pipeline.create(dai.node.ColorCamera)  # 创建右摄像头

    manip = pipeline.create(dai.node.ImageManip)  # 创建图像处理节点

    stereo = pipeline.create(dai.node.StereoDepth)  # 创建立体深度节点
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)  # 创建空间位置计算节点

    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)  # 创建YOLO检测网络

    imageOut = pipeline.create(dai.node.XLinkOut)  # 创建图像输出节点
    disparityOut = pipeline.create(dai.node.XLinkOut)  # 创建视差输出节点
    xoutNN = pipeline.create(dai.node.XLinkOut)  # 创建神经网络输出节点

    xoutSpatialData = pipeline.create(dai.node.XLinkOut)  # 创建空间数据输出节点
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)  # 创建空间计算配置输入节点

    # 设置输出流名称
    imageOut.setStreamName("image")
    disparityOut.setStreamName("disp")
    xoutNN.setStreamName("detections")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # 摄像头属性设置
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置分辨率
    left.setFps(30)  # 设置帧率
    left.setIspScale(1, 2)  # 设置ISP缩放
    right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置分辨率
    right.setFps(30)  # 设置帧率
    right.setIspScale(1, 2)  # 设置ISP缩放

    # 图像处理设置
    manip.initialConfig.setResize(W, H)  # 设置调整大小
    manip.initialConfig.setKeepAspectRatio(False)  # 不保持宽高比
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)  # 设置帧类型为BGR888p
    right.video.link(manip.inputImage)  # 连接右摄像头与图像处理节点

    # 设置摄像头的插槽
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # 左摄像头
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # 右摄像头

    # 深度计算设置
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)  # 设置预设模式
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)  # 设置中值滤波
    stereo.setLeftRightCheck(lr_check)  # 设置左右检查
    stereo.setExtendedDisparity(extended_disparity)  # 设置扩展视差
    stereo.setSubpixel(subpixel)  # 设置子像素
    stereo.setOutputSize(right.getIspWidth(), right.getIspHeight())  # 设置输出大小
    stereo.setDepthAlign(right.getBoardSocket())  # 设置深度对齐
    print(f"Depth aligner: {right.getBoardSocket()}")  # 打印深度对齐信息

    # 网络特定设置
    spatialDetectionNetwork.setBlob(model)  # 设置YOLO模型
    spatialDetectionNetwork.setConfidenceThreshold(0.5)  # 设置置信度阈值

    # YOLO特定参数
    spatialDetectionNetwork.setNumClasses(num_classes)  # 设置类别数量
    spatialDetectionNetwork.setCoordinateSize(4)  # 设置坐标大小
    spatialDetectionNetwork.setAnchors([])  # 设置锚点
    spatialDetectionNetwork.setAnchorMasks({})  # 设置锚点掩膜
    spatialDetectionNetwork.setIouThreshold(0.3)  # 设置IOU阈值

    # 空间特定参数
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)  # 设置边界框缩放因子
    spatialDetectionNetwork.setDepthLowerThreshold(lower_threshold)  # 设置深度下阈值
    spatialDetectionNetwork.setDepthUpperThreshold(upper_threshold)  # 设置深度上阈值
    spatialDetectionNetwork.setSpatialCalculationAlgorithm(calculation_algorithm)  # 设置空间计算算法

    # 配置设置
    config = dai.SpatialLocationCalculatorConfigData()  # 创建配置数据对象
    config.depthThresholds.lowerThreshold = lower_threshold  # 设置深度下阈值
    config.depthThresholds.upperThreshold = upper_threshold  # 设置深度上阈值
    config.calculationAlgorithm = calculation_algorithm  # 设置计算算法
    config.roi = dai.Rect(top_left, bottom_right)  # 设置感兴趣区域

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)  # 设置不等待消息
    spatialLocationCalculator.initialConfig.addROI(config)  # 添加感兴趣区域配置

    # 链接各个节点
    # right.isp.link(imageOut.input)  # 连接右摄像头的ISP与图像输出
    stereo.syncedRight.link(imageOut.input)  # 连接同步右摄像头的深度输出与图像输出
    manip.out.link(spatialDetectionNetwork.input)  # 连接图像处理输出与YOLO输入

    left.isp.link(stereo.left)  # 连接左摄像头ISP与立体深度左输入
    right.isp.link(stereo.right)  # 连接右摄像头ISP与立体深度右输入

    stereo.disparity.link(disparityOut.input)  # 连接立体深度的视差输出与视差输出
    stereo.depth.link(spatialDetectionNetwork.inputDepth)  # 连接深度输出与YOLO输入深度

    spatialDetectionNetwork.passthroughDepth.link(spatialLocationCalculator.inputDepth)  # 连接YOLO深度输出与空间位置计算输入
    spatialDetectionNetwork.out.link(xoutNN.input)  # 连接YOLO输出与神经网络输出

    spatialLocationCalculator.out.link(xoutSpatialData.input)  # 连接空间位置计算输出与空间数据输出

    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)  # 连接输入配置与空间计算节点

    return pipeline, stereo.initialConfig.getMaxDisparity()  # 返回管道和最大视差值


def angle_CBD_complement(B, C, D):
    # 假设检测框是左上角为A 顺时针顶点为A B D C的矩形
    #     A             B
    #      +-----------+
    #      |        /  |
    #      |      /    |
    #      |    /      |
    #      |   /       |
    #      +-----------+
    #     C             D
    # 矩形左上角和右下角坐标 计算向量

    v1 = (C[0] - B[0], C[1] - B[1])  # BC
    v2 = (D[0] - B[0], D[1] - B[1])  # BD

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.acos(cos_angle)

    # 计算余角
    complement_angle = 90 - math.degrees(angle)
    return complement_angle


def check_input(roi, frame, DELTA=5):
    """检查输入是否为ROI或点。如果是点，则转换为ROI"""
    # 如果输入是列表，则转换为numpy数组
    if isinstance(roi, list):
        roi = np.array(roi)

    # 限制点的范围，以免ROI超出帧范围
    if roi.shape in {(2,), (2, 1)}:  # 如果是点
        roi = np.hstack([roi, np.array([[-DELTA, -DELTA], [DELTA, DELTA]])])  # 扩展为ROI
    elif roi.shape in {(4,), (4, 1)}:  # 如果是四个坐标
        roi = np.array(roi)

    # 将ROI限制在帧的范围内
    roi.clip([DELTA, DELTA], [frame.shape[1] - DELTA, frame.shape[0] - DELTA])

    return roi / frame.shape[1::-1]  # 返回归一化后的ROI


def click_and_crop(event, x, y, flags, param):
    global ref_pt, click_roi

    # 当按下鼠标左键时，记录起始点
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt = [(x, y)]  # 初始化为列表，存储起始点

    # 当松开鼠标左键时，记录终点并完成矩形区域
    elif event == cv2.EVENT_LBUTTONUP:
        ref_pt.append((x, y))  # 追加终点坐标
        ref_pt = np.array(ref_pt)  # 将列表转换为NumPy数组
        click_roi = np.array([np.min(ref_pt, axis=0), np.max(ref_pt, axis=0)])  # 确保ROI的最小和最大值
        print(f"ROI Selected: {click_roi}")  # 打印选择的ROI


def run():
    global ref_pt, click_roi, calculation_algorithm, config
    # 连接到设备并启动流水线
    with dai.Device(dai.DeviceInfo("10.40.12.75")) as device:
        # 创建设备流水线和最大视差
        pipeline, maxDisparity = create_pipeline()
        device.startPipeline(pipeline)

        # 初始化RGB和深度图像帧、深度数据、检测结果
        frameRgb = None
        frameDisp = None
        depthDatas = []
        detections = []
        # 为不同类别生成随机颜色框
        bboxColors = np.random.default_rng().integers(256, size=(num_classes, 3)).tolist()
        step_size = 0.01  # ROI的移动步长
        new_config = False  # 是否需要新的配置

        ir_dot = 0.0  # 红外激光点投影仪的强度初始值

        # 配置显示窗口，trackbar调整RGB与深度数据的混合比例
        rgbWindowName = "image"
        depthWindowName = "depthQueueData"
        cv2.namedWindow(rgbWindowName)
        cv2.namedWindow(depthWindowName)

        # 设置鼠标点击回调函数，用于ROI选择

        print("使用WASD键移动ROI!")

        # 获取输入和输出队列
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        imageQueue = device.getOutputQueue("image")
        dispQueue = device.getOutputQueue("disp")
        spatialDataQueue = device.getOutputQueue("spatialData")
        detectQueue = device.getOutputQueue(name="detections")

        # 归一化边界框
        def frame_norm(frame, bbox):
            """NN 数据作为边界框位置，位于 <0..1> 范围内它们需要用帧宽度/高度进行归一化"""
            normVals = np.full(len(bbox), frame.shape[0])  # 用帧的高度填充数组
            normVals[::2] = frame.shape[1]  # 每隔一项使用帧的宽度
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)  # 归一化并转换为整数

        # 在帧上绘制文本
        def draw_text(frame, text, org, color=(255, 255, 255), thickness=1):
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA)  # 黑色阴影
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)  # 白色文本

        # 在帧上绘制矩形框
        def draw_rect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
            cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)  # 黑色阴影
            cv2.rectangle(frame, topLeft, bottomRight, color, thickness)  # 指定颜色的矩形

        # 在帧上绘制检测结果
        def draw_detection(frame, detections):
            for detection in detections:
                # 归一化边界框，转换为帧尺寸
                bbox = frame_norm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )
                # 绘制类别标签
                draw_text(
                    frame,
                    label_map[detection.label],  # 从标签映射中获取类别名称
                    (bbox[0] + 10, bbox[1] + 20),  # 标签位置
                )
                # 绘制置信度
                draw_text(
                    frame,
                    f"{detection.confidence:.2%}",  # 置信度格式化为百分比
                    (bbox[0] + 10, bbox[1] + 35),
                )
                # 绘制检测框
                draw_rect(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bboxColors[detection.label], 1)
                # 点坐标
                point_B = (bbox[2], bbox[1])  # 右上角
                point_C = (bbox[2], bbox[3])  # 右下角
                point_D = (bbox[0], bbox[3])  # 左下角
                angle_R = angle_CBD_complement(point_B, point_C, point_D)
                # 如果检测结果包含空间坐标，绘制X, Y, Z坐标
                if hasattr(detection, "spatialCoordinates"):
                    print(f"X: {int(detection.spatialCoordinates.x)} mm")
                    draw_text(
                        frame,
                        f"X: {int(detection.spatialCoordinates.x)} mm",  # X坐标
                        (bbox[0] + 10, bbox[1] + 50),
                    )
                    print(f"Y: {int(detection.spatialCoordinates.y)} mm")
                    draw_text(
                        frame,
                        f"Y: {int(detection.spatialCoordinates.y)} mm",  # Y坐标
                        (bbox[0] + 10, bbox[1] + 65),
                    )
                    print(f"Z: {int(detection.spatialCoordinates.z)} mm")
                    draw_text(
                        frame,
                        f"Z: {int(detection.spatialCoordinates.z)} mm",  # Z坐标
                        (bbox[0] + 10, bbox[1] + 80),
                    )
                    draw_text(
                        frame,
                        f"R: {int(angle_R)} degrees",  # R 角
                        (bbox[0] + 10, bbox[1] + 95),
                    )
                    print(f"R: {int(angle_R)} degrees")

        # 定义绘制空间位置的函数
        def draw_spatial_locations(frame, spatialLocations):
            # 遍历每一个空间位置数据
            for depthData in spatialLocations:
                # 获取深度数据中的ROI（感兴趣区域）
                roi = depthData.config.roi
                # 将ROI的坐标反归一化，转换为图像尺寸
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                xmin = int(roi.topLeft().x)  # ROI的左上角x坐标
                ymin = int(roi.topLeft().y)  # ROI的左上角y坐标
                xmax = int(roi.bottomRight().x)  # ROI的右下角x坐标
                ymax = int(roi.bottomRight().y)  # ROI的右下角y坐标

                # 在图像上绘制黑色和白色的矩形框
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 4)  # 黑色边框
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)  # 白色边框

                # 绘制X、Y、Z坐标值
                draw_text(frame, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20))
                draw_text(frame, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35))
                draw_text(frame, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50))

        # 定义帧率控制器
        fps = FPSHandler()

        # 主循环，检查设备是否关闭
        while not device.isClosed():
            # 尝试获取图像、深度、空间数据和检测数据
            imageData = imageQueue.tryGet()
            dispData = dispQueue.tryGet()
            spatialData = spatialDataQueue.tryGet()
            detData = detectQueue.tryGet()

            # 如果有空间数据，获取空间位置
            if spatialData is not None:
                depthDatas = spatialData.getSpatialLocations()

            # 如果有检测数据，处理帧率和检测结果
            if detData is not None:
                fps.tick("nn")
                detections = detData.detections

            # 如果有图像数据，绘制检测和空间位置，显示帧率
            if imageData is not None:
                frameRgb = imageData.getCvFrame()
                draw_detection(frameRgb, detections)
                draw_spatial_locations(frameRgb, depthDatas)
                fps.tick("image")
                fps.drawFps(frameRgb, "image")

                # 显示RGB图像
                cv2.imshow(rgbWindowName, frameRgb)

            # 如果有深度数据，绘制检测和空间位置，显示帧率
            if dispData is not None:
                frameDisp = dispData.getFrame()
                frameDisp = (frameDisp * (255 / maxDisparity)).astype(np.uint8)  # 深度数据归一化
                frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)  # 应用颜色映射
                frameDisp = np.ascontiguousarray(frameDisp)
                draw_detection(frameDisp, detections)
                draw_spatial_locations(frameDisp, depthDatas)
                fps.tick("dispData")
                fps.drawFps(frameDisp, "dispData")
                cv2.imshow(depthWindowName, frameDisp)

            # 当RGB和深度图像都收到时，检查用户输入的ROI
            if frameRgb is not None and frameDisp is not None and click_roi is not None:
                # 获取ROI的上下左右坐标
                (
                    [top_left.x, top_left.y],
                    [bottom_right.x, bottom_right.y],
                ) = check_input(click_roi, frameRgb)
                click_roi = None  # 清除ROI
                new_config = True  # 设定需要新的配置

            # 处理用户按键输入
            key = cv2.waitKey(1)
            if key == ord("q"):
                break  # 按'q'退出程序
            if key == ord("w"):
                if top_left.y - step_size >= 0:
                    top_left.y -= step_size
                    bottom_right.y -= step_size
                    new_config = True  # 设定新的ROI配置
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
            # 调整大小
            elif key == ord("z"):
                if bottom_right.y - step_size >= 0 and bottom_right.x - step_size >= 0:
                    bottom_right.y -= step_size
                    bottom_right.x -= step_size
                    new_config = True
            elif key == ord("x"):
                if bottom_right.x + step_size <= 1 and bottom_right.y + step_size <= 1:
                    bottom_right.y += step_size
                    bottom_right.x += step_size
                    new_config = True
            # 切换计算算法
            elif key == ord("1"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
                print("切换计算算法为MEAN!")
                new_config = True
            elif key == ord("2"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
                print("切换计算算法为MIN!")
                new_config = True
            elif key == ord("3"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MAX
                print("切换计算算法为MAX!")
                new_config = True
            elif key == ord("4"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
                print("切换计算算法为MODE!")
                new_config = True
            elif key == ord("5"):
                calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                print("切换计算算法为MEDIAN!")
                new_config = True

            # 调整IR激光点投影仪的强度
            elif key == ord(","):
                ir_dot -= 0.1
                ir_dot = max(0, ir_dot)  # 防止低于0
                if device.setIrLaserDotProjectorIntensity(ir_dot):
                    print(f"IR激光点投影仪强度设置为 {ir_dot}")
            elif key == ord("."):
                ir_dot += 0.1
                ir_dot = min(1, ir_dot)  # 防止超过1
                if device.setIrLaserDotProjectorIntensity(ir_dot):
                    print(f"IR激光点投影仪强度设置为 {ir_dot}")

            # 如果有新的配置，则发送新的ROI和计算算法配置
            if new_config:
                config.roi = dai.Rect(top_left, bottom_right)
                config.calculationAlgorithm = calculation_algorithm
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)  # 发送新的空间计算配置
                new_config = False  # 配置发送完成，重置标志


if __name__ == "__main__":
    ref_pt = None
    click_roi = None
    run()
