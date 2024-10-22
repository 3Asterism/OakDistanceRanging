#!/usr/bin/env python3

import depthai as dai  # 导入DepthAI库
import numpy as np  # 导入NumPy库，用于处理数组
import sys  # 导入系统库，用于命令行参数
from pathlib import Path  # 导入Path库，用于处理文件路径

# 连接设备，设备IP地址为169.254.1.222
with dai.Device(dai.DeviceInfo("169.254.1.222")) as device:
    # 构建相机校准文件的路径，名称为calib_<设备ID>.json
    calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())

    # 如果有命令行参数，则使用该参数作为校准文件路径
    if len(sys.argv) > 1:
        calibFile = sys.argv[1]

    # 读取设备的校准数据并保存为json文件
    calibData = device.readCalibration()
    calibData.eepromToJsonFile(calibFile)

    # 获取RGB相机的默认内参矩阵、宽度和高度
    M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
    print("RGB Camera Default intrinsics...")
    print(M_rgb)
    print(width)
    print(height)

    # 如果设备是OAK-1或BW1093OAK
    if "OAK-1" in calibData.getEepromData().boardName or "BW1093OAK" in calibData.getEepromData().boardName:
        # 获取分辨率为1280x720时的RGB相机内参矩阵
        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720))
        print("RGB Camera resized intrinsics...")
        print(M_rgb)

        # 获取RGB相机的畸变系数
        D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        print("RGB Distortion Coefficients...")
        # 打印畸变系数的名称和值
        [print(name + ": " + value) for (name, value) in
         zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
             [str(data) for data in D_rgb])]

        # 打印RGB相机的视场角(FOV)
        print(f'RGB FOV {calibData.getFov(dai.CameraBoardSocket.CAM_A)}')

    else:
        # 如果不是OAK-1或BW1093OAK设备，继续获取并打印RGB相机的默认内参
        M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
        print("RGB Camera Default intrinsics...")
        print(M_rgb)
        print(width)
        print(height)

        # 获取分辨率为3840x2160时的RGB相机内参
        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 3840, 2160))
        print("RGB Camera resized intrinsics... 3840 x 2160 ")
        print(M_rgb)

        # 获取分辨率为4056x3040时的RGB相机内参
        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 4056, 3040))
        print("RGB Camera resized intrinsics... 4056 x 3040 ")
        print(M_rgb)

        # 获取左相机的默认内参矩阵、宽度和高度
        M_left, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_B)
        print("LEFT Camera Default intrinsics...")
        print(M_left)
        print(width)
        print(height)

        # 获取分辨率为1280x720时左相机的内参矩阵
        M_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 720))
        print("LEFT Camera resized intrinsics...  1280 x 720")
        print(M_left)

        # 获取分辨率为1280x720时右相机的内参矩阵
        M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 1280, 720))
        print("RIGHT Camera resized intrinsics... 1280 x 720")
        print(M_right)

        # 获取左相机的畸变系数
        D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
        print("LEFT Distortion Coefficients...")
        [print(name + ": " + value) for (name, value) in zip(
            ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
            [str(data) for data in D_left])]

        # 获取右相机的畸变系数
        D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
        print("RIGHT Distortion Coefficients...")
        [print(name + ": " + value) for (name, value) in zip(
            ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
            [str(data) for data in D_right])]

        # 打印RGB和Mono（单目）相机的视场角(FOV)
        print(
            f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.CAM_A)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.CAM_B)}")

        # 获取立体校正旋转矩阵
        R1 = np.array(calibData.getStereoLeftRectificationRotation())
        R2 = np.array(calibData.getStereoRightRectificationRotation())

        # 计算左相机的立体校正矩阵
        H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
        print("LEFT Camera stereo rectification matrix...")
        print(H_left)

        # 计算右相机的立体校正矩阵
        H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))
        print("RIGHT Camera stereo rectification matrix...")
        print(H_right)

        # 获取左相机相对于右相机的外参矩阵（变换矩阵）
        lr_extrinsics = np.array(
            calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C))
        print("Transformation matrix of where left Camera is W.R.T right Camera's optical center")
        print(lr_extrinsics)

        # 获取左相机相对于RGB相机的外参矩阵
        l_rgb_extrinsics = np.array(
            calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A))
        print("Transformation matrix of where left Camera is W.R.T RGB Camera's optical center")
        print(l_rgb_extrinsics)
