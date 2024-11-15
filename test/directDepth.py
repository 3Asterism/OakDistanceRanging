import math

import cv2
import depthai as dai
from ultralytics import YOLO


def calculate_rotation_angle(box_points, head_point):
    """
    计算旋转角度并确定旋转方向
    box_points: 边界框的四个角点坐标 (x1,y1, x2,y2)
    head_point: 头部关键点坐标 (x,y)
    返回: 带符号的角度（正值表示顺时针，负值表示逆时针）
    """
    # 计算边界框的中心点
    center_x = (box_points[0] + box_points[2]) / 2
    center_y = (box_points[1] + box_points[3]) / 2

    # 计算头部到中心点的向量
    direction_vector = [
        head_point[0] - center_x,
        head_point[1] - center_y
    ]

    # 计算向量与垂直向上方向的夹角
    angle = math.degrees(math.atan2(direction_vector[0], -direction_vector[1]))

    # 确保角度在-180到180度之间
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360

    return angle


# 加载YOLOv11模型
model = YOLO("best.pt")

# 创建 DepthAI 管道
pipeline = dai.Pipeline()

# 设置摄像头节点
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
cam_rgb.setFps(30)
cam_rgb.setIspScale(1, 2)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_B)

# 设置输出节点
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

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
            if r.boxes is not None and r.keypoints is not None:
                boxes_data = r.boxes.data
                keypoints_data = r.keypoints.data

                if len(boxes_data) > 0:
                    for obj_idx, (box, keypoints) in enumerate(zip(boxes_data, keypoints_data)):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = float(box[4])

                        # 获取头部关键点
                        if len(keypoints) > 0:
                            head_point = keypoints[0]
                            head_x, head_y = map(int, head_point[:2])

                            # 计算旋转角度
                            rotation_angle = calculate_rotation_angle(
                                [x1, y1, x2, y2],
                                [head_x, head_y]
                            )

                            # 打印信息
                            print(f"\n物体 #{obj_idx + 1}:")
                            print(f"边界框坐标: 左上({x1}, {y1}), 右下({x2}, {y2})")
                            print(f"置信度: {conf:.2f}")
                            print(f"头部坐标: ({head_x}, {head_y})")
                            print(f"旋转角度: {rotation_angle:.2f}度")

                            # 绘制边界框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            # 绘制标签
                            label = f"#{obj_idx + 1} Rot: {rotation_angle:.1f}°"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                            # 绘制头部关键点
                            cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)

                            # 绘制方向指示线
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            # 使用极坐标方式计算箭头终点
                            line_length = 50
                            angle_rad = math.radians(rotation_angle)
                            end_x = int(center_x + line_length * math.sin(angle_rad))
                            end_y = int(center_y - line_length * math.cos(angle_rad))

                            # 绘制方向线
                            cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
                            # 绘制中心点
                            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

                            # 绘制其他关键点
                            for kp in keypoints[1:]:
                                x, y = map(int, kp[:2])
                                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # 显示结果
        cv2.imshow("YOLOv11-Pose Detection with Rotation", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()