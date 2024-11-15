#!/usr/bin/env python3

import math
import pathlib
import queue
import threading

import cv2
import depthai as dai

from calc import HostSpatialsCalc
from utility import TextHelper

# 将 pathlib.PosixPath 设置为 pathlib.WindowsPath 以解决 Windows 系统上的路径问题
pathlib.PosixPath = pathlib.WindowsPath
# 最小深度设置，视差范围加倍（从95到190）：
extended_disparity = True
# 更好地处理长距离的精度，分数视差32级：
subpixel = True
# 更好地处理遮挡：
lr_check = True


def create_pipeline():
    pipeline = dai.Pipeline()

    # 定义深度相机和 RGB 相机
    monoLeft = pipeline.create(dai.node.ColorCamera)
    monoRight = pipeline.create(dai.node.ColorCamera)
    camRgb = pipeline.create(dai.node.ColorCamera)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)  # 创建空间计算配置输入节点
    xoutNN = pipeline.create(dai.node.XLinkOut)  # 创建神经网络输出节点

    # 摄像头属性设置
    monoLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置分辨率
    monoLeft.setFps(24)  # 设置帧率
    monoLeft.setIspScale(1, 2)  # 设置ISP缩放
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)  # 创建空间位置计算节点

    # 配置右单目相机
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置分辨率
    monoRight.setFps(24)  # 设置帧率
    monoRight.setIspScale(1, 2)  # 设置ISP缩放

    # 配置深度相机
    stereo = pipeline.create(dai.node.StereoDepth)
    # 深度计算设置
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)  # 设置预设模式
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)  # 设置中值滤波
    stereo.setLeftRightCheck(lr_check)  # 设置左右检查
    stereo.setExtendedDisparity(extended_disparity)  # 设置扩展视差
    stereo.setSubpixel(subpixel)  # 设置子像素
    stereo.setOutputSize(monoRight.getIspWidth(), monoRight.getIspHeight())  # 设置输出大小
    stereo.setDepthAlign(monoRight.getBoardSocket())  # 设置深度对齐
    print(f"Depth aligner: {monoRight.getBoardSocket()}")  # 打印深度对齐信息

    imageOut = pipeline.create(dai.node.XLinkOut)  # 创建图像输出节点
    # 链接各个节点
    stereo.syncedRight.link(imageOut.input)  # 连接同步右摄像头的深度输出与图像输出
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)  # 创建YOLO检测网络

    monoLeft.isp.link(stereo.left)  # 连接左摄像头ISP与立体深度左输入
    monoRight.isp.link(stereo.right)  # 连接右摄像头ISP与立体深度右输入

    stereo.depth.link(spatialDetectionNetwork.inputDepth)  # 连接深度输出与YOLO输入深度
    spatialDetectionNetwork.passthroughDepth.link(spatialLocationCalculator.inputDepth)  # 连接YOLO深度输出与空间位置计算输入
    spatialDetectionNetwork.out.link(xoutNN.input)  # 连接YOLO输出与神经网络输出

    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)  # 连接输入配置与空间计算节点

    # 配置彩色相机
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(3)

    # 创建深度输出
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    # 创建 RGB 输出
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)

    return pipeline, stereo


def load_model(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)  # Load the YOLOv8 model
    return model


class DepthAIApp:
    def __init__(self, model_path, device_ip):
        self.model = load_model(model_path)
        self.pipeline, self.stereo = create_pipeline()
        self.device_ip = device_ip
        # 方形检测区域的大小
        self.delta = 2
        self.text_helper = TextHelper()
        self.frame_queue = queue.Queue(maxsize=5)  # 用于存储待处理帧的队列

    def process_detection(self, detection, depthData, hostSpatials, rgbFrame):
        x1, y1, x2, y2, conf, cls = detection
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        # Calculate spatial coordinates from depth frame
        spatials, centroid = hostSpatials.calc_spatials(depthData, (x, y))  # centroid == x/y in our case

        # 输出检测框四个端点的 xy 坐标
        print(f"Detection box corners: ({x1}, {y1}), ({x2}, {y1}), ({x2}, {y2}), ({x1}, {y2})")

        print("x: " + str(x) + "X: " + (
            "{:.5f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"))
        print("y: " + str(y) + "Y: " + (
            "{:.5f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"))
        print("Z: " + ("{:.5f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"))

        # 计算检测框对角线的角度
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        print(f"Diagonal angle: {angle:.4f} degrees")

        # 在 RGB 帧上绘制矩形框和空间坐标信息
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        self.text_helper.rectangle(rgbFrame, pt1, pt2)

        return rgbFrame, all(not math.isnan(spatials[key]) for key in ['x', 'y', 'z'])

    def inference_thread(self, hostSpatials):
        while True:
            depthData, rgbFrame = self.frame_queue.get()
            if depthData is None or rgbFrame is None:
                break

            # YOLOv8 推理
            results = self.model(rgbFrame)[0]  # 获取推理结果

            # 提取检测结果 这里是cpu 可以改成gpu推理
            detections = results.boxes.data.cpu().numpy()  # 提取检测框数据

            all_spatial_data_obtained = True
            for detection in detections:
                rgbFrame, spatial_data_obtained = self.process_detection(detection, depthData, hostSpatials, rgbFrame)
                if not spatial_data_obtained:
                    all_spatial_data_obtained = False

            cv2.imshow("rgb-depth", cv2.resize(rgbFrame, (1920, 1080)))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            # 如果所有检测对象的空间数据都已获取，退出循环
            if all_spatial_data_obtained:
                break

    def start_device(self):
        # 连接到设备并启动管道
        with dai.Device(self.pipeline, dai.DeviceInfo(self.device_ip)) as device:
            depthQueue = device.getOutputQueue(name="depth")
            rgbQueue = device.getOutputQueue(name="rgb")

            hostSpatials = HostSpatialsCalc(device)
            hostSpatials.setDeltaRoi(self.delta)

            # 启动推理线程
            threading.Thread(target=self.inference_thread, args=(hostSpatials,), daemon=True).start()

            while True:
                depthData = depthQueue.get()
                rgbData = rgbQueue.get()
                rgbFrame = rgbData.getCvFrame()

                # 将深度数据和 RGB 帧放入队列
                self.frame_queue.put((depthData, rgbFrame))

                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.frame_queue.put((None, None))  # 终止推理线程
                    break


if __name__ == "__main__":
    model_file_path = r'best.pt'
    device_ip_address = "10.40.4.1"
    app = DepthAIApp(model_file_path, device_ip_address)
    app.start_device()
