import math
import cv2
import depthai as dai
from ultralytics import YOLO


def calculate_center(box_points):
    """
    计算边界框的中心点
    box_points: 边界框的四个角点坐标 (x1, y1, x2, y2)
    返回: 中心点坐标 (center_x, center_y)
    """
    center_x = (box_points[0] + box_points[2]) / 2
    center_y = (box_points[1] + box_points[3]) / 2
    return center_x, center_y


# 加载YOLOv11模型
model = YOLO("bestBox.pt")

# 创建 DepthAI 管道
pipeline = dai.Pipeline()

# 设置摄像头节点，保持原始分辨率 (1280x800)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # 设置为800P分辨率 (1280x800)
cam_rgb.setFps(30)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# 创建图像处理节点 (ImageManip)，调整输出分辨率为 1280x800
manip = pipeline.create(dai.node.ImageManip)
manip.setMaxOutputFrameSize(3072000)  # 设置最大输出帧大小为3MB
manip.setResize(1200,800)
cam_rgb.preview.link(manip.inputImage)  # 连接摄像头和图像处理节点

# 设置输出节点
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
manip.out.link(xout_rgb.input)  # 连接图像处理输出到 XLinkOut

# 连接到设备并开始管道
with dai.Device(pipeline, dai.DeviceInfo("10.40.4.1")) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = rgb_queue.get()
        frame = in_rgb.getCvFrame()

        # 使用YOLOv11进行预测
        results = model(frame)

        # 处理每个检测结果
        for r in results:
            if r.boxes is not None:
                boxes_data = r.boxes.data

                if len(boxes_data) > 0:
                    for obj_idx, box in enumerate(boxes_data):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = float(box[4])

                        # 计算边界框的中心点
                        center_x, center_y = calculate_center([x1, y1, x2, y2])

                        # 打印信息
                        print(f"\n物体 #{obj_idx + 1}:")
                        print(f"边界框坐标: 左上({x1}, {y1}), 右下({x2}, {y2})")
                        print(f"置信度: {conf:.2f}")
                        print(f"中心点坐标: ({int(center_x)}, {int(center_y)})")

                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # 绘制标签
                        label = f"#{obj_idx + 1}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        # 绘制中心点
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)

        # 显示结果
        cv2.imshow("YOLOv11 Detection with Centers", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
