import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from math import sqrt


# 读取相机标定参数
def load_calibration(file_path):
    with open(file_path, 'r') as file:
        calibration_data = yaml.safe_load(file)
        camera_matrix = np.array(calibration_data['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(calibration_data['dist_coeff']['data'])
    return camera_matrix, dist_coeffs


# 加载标定参数
camera_matrix, dist_coeffs = load_calibration('../script/data/calibration_danmu.yaml')

# 加载YOLOv8模型
model = YOLO(r'C:\dataSet\result\weights\bestHead.pt')

# 初始化相机
cap = cv2.VideoCapture(0)  # 使用第一个摄像头

# 已知物体的实际长度（以米为单位）
l_known = 0.078  # 物体的实际对角线长度

# 实时视频流处理
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    # 校正图像
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

    # YOLOv8 进行目标检测
    results = model.predict(frame_undistorted, save=False, conf=0.5)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes.xywh.cpu().numpy()

    for box in boxes:
        x_center, y_center, pixel_width, pixel_height = box.tolist()
        x1 = int(x_center - pixel_width / 2)
        y1 = int(y_center - pixel_height / 2)
        x2 = int(x_center + pixel_width / 2)
        y2 = int(y_center + pixel_height / 2)

        # 计算检测框的对角线长度
        pixel_diagonal = sqrt(pixel_width ** 2 + pixel_height ** 2)

        # 焦距
        f = camera_matrix[1, 1]

        # 根据物体的实际对角线长度和检测框的对角线长度计算距离
        distance = (l_known * f) / pixel_diagonal

        # 在框上方显示距离
        cv2.putText(annotated_frame, f'Distance: {distance:.3f}m', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示检测结果
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
