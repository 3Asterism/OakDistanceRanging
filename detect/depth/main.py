from __future__ import annotations

from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import math
from ultralytics import YOLO

lower_threshold = 30  # 最小深度阈值，单位为毫米
upper_threshold = 150_0  # 最大深度阈值，单位为毫米

num_classes = 1  # 类别数量

# 加载模型文件
blob = Path(__file__).parent.parent.joinpath("model/best.blob")
model = YOLO("model/best.pt")  # 创建OpenVINO模型对象
dim = next(iter(model.networkInputs.values())).dims  # 获取输入维度
W, H = dim[:2]  # 获取宽度和高度

# 获取输出名称和张量
output_name, output_tenser = next(iter(model.networkOutputs.items()))

# 标签映射
label_map = [
    "head"
]

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

    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)  # 创建空间计算配置输入节点

    # 设置输出流名称
    imageOut.setStreamName("image")
    disparityOut.setStreamName("disp")
    xoutNN.setStreamName("detections")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # 摄像头属性设置
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置分辨率
    left.setFps(24)  # 设置帧率
    left.setIspScale(1, 2)  # 设置ISP缩放
    right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置分辨率
    right.setFps(24)  # 设置帧率
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
    stereo.syncedRight.link(imageOut.input)  # 连接同步右摄像头的深度输出与图像输出
    manip.out.link(spatialDetectionNetwork.input)  # 连接图像处理输出与YOLO输入

    left.isp.link(stereo.left)  # 连接左摄像头ISP与立体深度左输入
    right.isp.link(stereo.right)  # 连接右摄像头ISP与立体深度右输入

    stereo.disparity.link(disparityOut.input)  # 连接立体深度的视差输出与视差输出
    stereo.depth.link(spatialDetectionNetwork.inputDepth)  # 连接深度输出与YOLO输入深度

    spatialDetectionNetwork.passthroughDepth.link(spatialLocationCalculator.inputDepth)  # 连接YOLO深度输出与空间位置计算输入
    spatialDetectionNetwork.out.link(xoutNN.input)  # 连接YOLO输出与神经网络输出

    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)  # 连接输入配置与空间计算节点

    return pipeline, stereo.initialConfig.getMaxDisparity()  # 返回管道和最大视差值


def run():
    global ref_pt, click_roi, calculation_algorithm, config
    # 连接到设备并启动流水线
    with dai.Device(dai.DeviceInfo("169.254.1.222")) as device:
        # 归一化边界框
        def frame_norm(frame, bbox):
            """NN 数据作为边界框位置，位于 <0..1> 范围内它们需要用帧宽度/高度进行归一化"""
            normVals = np.full(len(bbox), frame.shape[0])  # 用帧的高度填充数组
            normVals[::2] = frame.shape[1]  # 每隔一项使用帧的宽度
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)  # 归一化并转换为整数

        # 创建设备流水线和最大视差
        pipeline, maxDisparity = create_pipeline()
        device.startPipeline(pipeline)

        # 初始化RGB和深度图像帧、深度数据、检测结果
        frameRgb = None
        frameDisp = None
        detections = []

        # 获取输入和输出队列
        imageQueue = device.getOutputQueue("image")
        dispQueue = device.getOutputQueue("disp")
        detectQueue = device.getOutputQueue(name="detections")

        # 在帧上绘制检测结果
        def draw_detection(frame, detections):
            for detection in detections:
                # 归一化边界框，转换为帧尺寸
                bbox = frame_norm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )
                # 点坐标
                point_B = (bbox[2], bbox[1])  # 右上角
                point_C = (bbox[2], bbox[3])  # 右下角
                point_D = (bbox[0], bbox[3])  # 左下角
                angle_R = angle_CBD_complement(point_B, point_C, point_D)
                # 如果检测结果包含空间坐标，绘制X, Y, Z坐标
                if hasattr(detection, "spatialCoordinates") and int(detection.spatialCoordinates.z) > 30:
                    print(f"X: {int(detection.spatialCoordinates.x)} mm")
                    print(f"Y: {int(detection.spatialCoordinates.y)} mm")
                    print(f"Z: {int(detection.spatialCoordinates.z)} mm")
                    print(f"R: {int(angle_R)} degrees")
        # 主循环，检查设备是否关闭
        while not device.isClosed():
            # 尝试获取图像、深度、空间数据和检测数据
            imageData = imageQueue.tryGet()
            dispData = dispQueue.tryGet()
            detData = detectQueue.tryGet()

            # 如果有检测数据，处理帧率和检测结果
            if detData is not None:
                detections = detData.detections

            # 如果有图像数据，绘制检测和空间位置，显示帧率
            if imageData is not None:
                frameRgb = imageData.getCvFrame()
                draw_detection(frameRgb, detections)

            # 如果有深度数据，绘制检测和空间位置，显示帧率
            if dispData is not None:
                frameDisp = dispData.getFrame()
                frameDisp = (frameDisp * (255 / maxDisparity)).astype(np.uint8)  # 深度数据归一化
                frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)  # 应用颜色映射
                frameDisp = np.ascontiguousarray(frameDisp)
                draw_detection(frameDisp, detections)


if __name__ == "__main__":
    ref_pt = None
    click_roi = None
    run()
