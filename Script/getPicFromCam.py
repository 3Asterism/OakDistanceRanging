import cv2
import os
import time

# 创建保存图像的文件夹
save_dir = 'pic'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 打开相机（通常0是内置摄像头，1是外接摄像头）
cap = cv2.VideoCapture(0)

# 检查摄像头是否打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化图像计数器
img_counter = 0
total_images = 50  # 设置总共截图次数
interval = 2  # 设置截图时间间隔（秒）
last_capture_time = time.time()  # 记录上一次截图的时间

print(f"每 {interval} 秒自动保存一次图像，保存 {total_images} 次后退出")

while img_counter < total_images:
    # 读取相机中的每一帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，退出...")
        break

    # 实时显示当前帧
    cv2.imshow('Camera', frame)

    # 获取当前时间
    current_time = time.time()

    # 如果时间间隔大于等于 2 秒，则保存图像
    if current_time - last_capture_time >= interval:
        img_name = os.path.join(save_dir, f"image_{img_counter:03}.png")
        cv2.imwrite(img_name, frame)
        print(f"保存图像到 {img_name}")
        img_counter += 1
        last_capture_time = current_time  # 更新最后截图的时间

    # 检测是否有键盘输入 'q' 键提前退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("手动退出程序")
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
