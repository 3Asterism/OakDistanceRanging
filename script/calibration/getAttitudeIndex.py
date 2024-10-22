import cv2
import numpy as np
import depthai as dai
import DobotDllType as dType
import time
import json

# Dobot机械臂连接设置
api = dType.load()
CON_STR = dType.DobotConnect.DobotConnect_NoError
state = dType.ConnectDobot(api, "", 115200)[0]

# 相机标定板设置
chessboard_size = (7, 7)
square_size = 0.024
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# 用于存储机械臂末端位置和相机图像位姿数据
end_effector_poses = []
camera_poses = []

# 连接DepthAI设备，获取左相机
with dai.Device(dai.DeviceInfo("169.254.1.222")) as device:
    calibData = device.readCalibration()
    pipeline = dai.Pipeline()

    cam_left = pipeline.createColorCamera()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)

    xout_video = pipeline.createXLinkOut()
    xout_video.setStreamName("video")
    cam_left.video.link(xout_video.input)

    device.startPipeline(pipeline)
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    for i in range(10):
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, rHead, 1)
        time.sleep(2)

        pos = dType.GetPose(api)
        end_effector_poses.append(pos[:4])

        in_frame = video_queue.get()
        frame = in_frame.getCvFrame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvec, tvec = cv2.solvePnP(objp, corners, M_left, None)

            if ret:
                camera_poses.append((rvec, tvec))

            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            cv2.imshow('img', frame)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

# 保存位姿数据到JSON文件
with open('poses_data.json', 'w') as f:
    json.dump({'end_effector_poses': end_effector_poses, 'camera_poses': camera_poses}, f)

# 关闭机械臂连接
dType.DisconnectDobot(api)
