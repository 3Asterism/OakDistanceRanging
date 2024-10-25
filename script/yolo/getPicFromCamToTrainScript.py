import cv2
import depthai as dai
import os
import uuid

"""通过摄像头拍照获得训练数据"""

# 创建Pipeline
pipeline = dai.Pipeline()

# 定义RGB摄像头
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.LEFT)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 创建输出
xoutVideo = pipeline.createXLinkOut()
xoutVideo.setStreamName("video")
camRgb.video.link(xoutVideo.input)

# 连接到设备并启动Pipeline
with dai.Device(pipeline, dai.DeviceInfo("169.254.1.222")) as device:
    # 获取视频输出队列
    videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    # 创建存储截图的文件夹
    save_path = r"C:\dataSet\keypoint\valPic"
    os.makedirs(save_path, exist_ok=True)

    while True:
        # 从队列中获取视频帧
        video_frame = videoQueue.get()
        frame = video_frame.getCvFrame()

        # 显示视频帧
        cv2.imshow("RGB Camera", frame)

        # 检查键盘输入
        key = cv2.waitKey(1)
        if key == ord('j'):
            # 按下 "j" 键时截图并保存
            filename = os.path.join(save_path, f"{uuid.uuid4().hex}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved to {filename}")
        elif key == 27:  # 按下 "Esc" 键退出
            break

    cv2.destroyAllWindows()
