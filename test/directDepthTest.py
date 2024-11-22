import math
import cv2
import depthai as dai
from ultralytics import YOLO


def detect_poses(frame, model):
    """
    对输入帧进行姿态检测,返回检测到的物体信息
    """

    def determine_diagonal_and_points(box_points, head_point):
        """
        确定使用哪条对角线并计算相关点的坐标
        box_points: 边界框坐标 (x1,y1,x2,y2)
        head_point: 头部关键点坐标 (x,y)
        返回:
            is_left_diagonal: bool, 是否是左对角线
            tail_point: tuple, 尾部坐标
            diagonal_mid: tuple, 对角线中点坐标
            grip_point: tuple, 夹取点坐标
        """
        x1, y1, x2, y2 = box_points
        head_x, head_y = head_point

        # 计算两条对角线端点
        left_diagonal = [(x1, y1), (x2, y2)]  # 左上到右下
        right_diagonal = [(x2, y1), (x1, y2)]  # 右上到左下

        # 计算头部点到两条对角线的距离
        def point_to_line_distance(point, line_start, line_end):
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end

            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            return numerator / denominator if denominator != 0 else float('inf')

        dist_to_left = point_to_line_distance((head_x, head_y), left_diagonal[0], left_diagonal[1])
        dist_to_right = point_to_line_distance((head_x, head_y), right_diagonal[0], right_diagonal[1])

        # 选择距离更近的对角线
        is_left_diagonal = dist_to_left < dist_to_right
        chosen_diagonal = left_diagonal if is_left_diagonal else right_diagonal

        # 计算头部点到对角线两端点的距离
        dist_to_start = math.sqrt((head_x - chosen_diagonal[0][0]) ** 2 + (head_y - chosen_diagonal[0][1]) ** 2)
        dist_to_end = math.sqrt((head_x - chosen_diagonal[1][0]) ** 2 + (head_y - chosen_diagonal[1][1]) ** 2)

        # 头部距离哪个端点近，另一个端点就是尾部
        tail_point = chosen_diagonal[1] if dist_to_start < dist_to_end else chosen_diagonal[0]

        # 计算对角线中点
        diagonal_mid = (
            int((chosen_diagonal[0][0] + chosen_diagonal[1][0]) / 2),
            int((chosen_diagonal[0][1] + chosen_diagonal[1][1]) / 2)
        )

        # 计算尾部和对角线中点的中点作为夹取点
        grip_point = (
            int((tail_point[0] + diagonal_mid[0]) / 2),
            int((tail_point[1] + diagonal_mid[1]) / 2)
        )

        return is_left_diagonal, tail_point, diagonal_mid, grip_point

    def calculate_rotation_angle(head_point, tail_point):
        """计算旋转角度"""
        dx = head_point[0] - tail_point[0]
        dy = head_point[1] - tail_point[1]

        angle = math.degrees(math.atan2(dx, -dy))
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360

        return angle

    # 存储检测结果
    detections = []

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

                        # 确定对角线类型和相关点坐标
                        is_left_diagonal, tail_point, diagonal_mid, grip_point = determine_diagonal_and_points(
                            [x1, y1, x2, y2],
                            [head_x, head_y]
                        )

                        # 计算旋转角度
                        rotation_angle = calculate_rotation_angle(
                            [head_x, head_y],
                            tail_point
                        )

                        # 存储检测结果
                        detection = {
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'angle': rotation_angle,
                            'confidence': conf,
                            'box': (x1, y1, x2, y2),
                            'head_point': (head_x, head_y),
                            'tail_point': tail_point,
                            'diagonal_mid': diagonal_mid,
                            'grip_point': grip_point,
                            'is_left_diagonal': is_left_diagonal
                        }
                        detections.append(detection)

                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # 绘制对角线
                        if is_left_diagonal:
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)  # 左对角线
                        else:
                            cv2.line(frame, (x2, y1), (x1, y2), (0, 255, 255), 1)  # 右对角线

                        # 绘制各个关键点
                        cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)  # 头部-绿色
                        cv2.circle(frame, tail_point, 5, (0, 0, 255), -1)  # 尾部-红色
                        cv2.circle(frame, diagonal_mid, 4, (255, 165, 0), -1)  # 对角线中点-橙色
                        cv2.circle(frame, grip_point, 6, (255, 0, 255), -1)  # 夹取点-紫色

                        # 绘制标签
                        diagonal_type = "左对角线" if is_left_diagonal else "右对角线"
                        label = f"#{obj_idx + 1} Rot: {rotation_angle:.1f}° ({diagonal_type})"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        # 画出夹取点到其他点的连线
                        cv2.line(frame, tail_point, diagonal_mid, (128, 128, 128), 1)  # 灰色连线
                        cv2.line(frame, grip_point, tail_point, (255, 192, 203), 1)  # 粉色连线到尾部
                        cv2.line(frame, grip_point, diagonal_mid, (255, 192, 203), 1)  # 粉色连线到对角线中点

    return detections


def main():
    # 加载YOLOv11模型
    model = YOLO(r"C:\Users\84238\PycharmProjects\pipetteDetect\test\best.pt")

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

            # 使用detect_poses函数进行检测
            detections = detect_poses(frame, model)

            # 打印检测结果
            for obj_idx, det in enumerate(detections):
                print(f"\n物体 #{obj_idx + 1}:")
                print(f"边界框坐标: 左上{det['box'][:2]}, 右下{det['box'][2:]}")
                print(f"置信度: {det['confidence']:.2f}")
                print(f"头部坐标: {det['head_point']}")
                print(f"尾部坐标: {det['tail_point']}")
                print(f"对角线中点: {det['diagonal_mid']}")
                print(f"夹取点: {det['grip_point']}")
                print(f"使用对角线: {'左对角线' if det['is_left_diagonal'] else '右对角线'}")
                print(f"旋转角度: {det['angle']:.2f}度")

            # 显示结果
            cv2.imshow("YOLOv11-Pose Detection with Rotation", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
