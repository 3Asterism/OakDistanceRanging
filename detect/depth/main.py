#!/usr/bin/env python3

import concurrent.futures
import math
import pathlib
import time

import cv2
import depthai as dai

from calc import HostSpatialsCalc
from utility import TextHelper

# 将 pathlib.PosixPath 设置为 pathlib.WindowsPath 以解决 Windows 系统上的路径问题
pathlib.PosixPath = pathlib.WindowsPath


def create_pipeline():
    # 创建深度AI管道
    pipeline = dai.Pipeline()

    # 定义左右单目相机
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    # 设置左右单目相机的分辨率和连接
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

    # 将左右单目相机连接到立体相机
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # 创建深度输出
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    # 创建左单目相机输出，用于YOLO推理
    xoutMonoLeft = pipeline.create(dai.node.XLinkOut)
    xoutMonoLeft.setStreamName("mono_left")
    monoLeft.out.link(xoutMonoLeft.input)

    return pipeline, stereo


def load_model(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)  # 加载 YOLOv8 模型
    return model


class DepthAIApp:
    def __init__(self, model_path, device_ip):
        self.model = load_model(model_path)
        self.pipeline, self.stereo = create_pipeline()
        self.device_ip = device_ip
        # 方形检测区域的大小
        self.delta = 2
        self.text_helper = TextHelper()

    def process_detection(self, detection, depthData, hostSpatials, monoLeftFrame):
        x1, y1, x2, y2, conf, cls = detection
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        # 从深度帧计算空间坐标
        spatials, centroid = hostSpatials.calc_spatials(depthData, (x, y))

        # 在单目相机的帧上绘制矩形框和空间坐标信息
        self.text_helper.rectangle(monoLeftFrame, (x - self.delta, y - self.delta), (x + self.delta, y + self.delta))
        self.text_helper.putText(monoLeftFrame, "X: " + (
            "{:.5f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                                 (x + 10, y + 20))
        self.text_helper.putText(monoLeftFrame, "Y: " + (
            "{:.5f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                                 (x + 10, y + 35))
        self.text_helper.putText(monoLeftFrame, "Z: " + (
            "{:.5f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                                 (x + 10, y + 50))
        return monoLeftFrame

    def start_device(self):
        while True:
            try:
                # 连接到设备并启动管道
                with dai.Device(self.pipeline, dai.DeviceInfo(self.device_ip)) as device:
                    depthQueue = device.getOutputQueue(name="depth")
                    monoLeftQueue = device.getOutputQueue(name="mono_left")

                    hostSpatials = HostSpatialsCalc(device)
                    hostSpatials.setDeltaRoi(self.delta)

                    while True:
                        depthData = depthQueue.get()
                        monoLeftData = monoLeftQueue.get()

                        # 获取左单目相机帧并转换为彩色图像
                        monoLeftFrame = monoLeftData.getCvFrame()
                        monoLeftFrame = cv2.cvtColor(monoLeftFrame, cv2.COLOR_GRAY2BGR)  # 转换为三通道图像

                        # YOLOv8 推理
                        results = self.model(monoLeftFrame)[0]  # 获取推理结果

                        # 提取检测结果
                        detections = results.boxes.data.cpu().numpy()  # 提取检测框数据

                        # 使用多线程处理每个检测结果
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = [
                                executor.submit(self.process_detection, detection, depthData, hostSpatials, monoLeftFrame)
                                for detection in detections]
                            for future in concurrent.futures.as_completed(futures):
                                monoLeftFrame = future.result()

                        # 显示推理结果
                        cv2.imshow("mono-left-depth", monoLeftFrame)

                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            return

            except Exception as e:
                print(f"设备连接失败: {e}")
                print("5秒后重试...")
                time.sleep(5)


if __name__ == "__main__":
    model_file_path = r'C:\dataSet\result\weights\bestHead.pt'
    device_ip_address = "169.254.1.222"
    app = DepthAIApp(model_file_path, device_ip_address)
    app.start_device()
