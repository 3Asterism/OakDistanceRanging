import cv2
import numpy as np
import glob
import os
import yaml

# 相机标定
# 设置棋盘格w和h方向的角点数量
w_corners = 11  # 改！数棋盘格宽有多少个格子，然后减一
h_corners = 8  # 改！数棋盘格高有多少个格子，然后减一
# 设置图像路径
images = glob.glob('pic/*.png')  # 改！改成自己存放图片的路径

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((w_corners * h_corners, 3), np.float32)
objp[:, :2] = np.mgrid[0:w_corners, 0:h_corners].T.reshape(-1, 2)
objp = objp * 20  # 改！这里的21是一个格子的长度，单位是mm

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

i = 0


def save_calibration_to_yaml(file_path, cameraMatrix_l, distCoeffs_l):
    data = {
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_l.flatten().tolist()
        },
        'dist_coeff': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_l.flatten().tolist()
        }
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Calibration parameters saved to {file_path}")


for fname in images:
    if not os.path.exists(fname):
        print(f"文件不存在: {fname}")
        continue

    try:
        with open(fname, 'rb') as f:
            print(f"文件正常读取: {fname}")
    except Exception as e:
        print(f"无法读取文件: {fname}, 错误: {e}")
        continue

    img = cv2.imread(fname)
    if img is None:
        print(f"OpenCV 无法读取文件: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (w_corners, h_corners), None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (w_corners, h_corners), corners, ret)
        i += 1

        new_size = (1280, 800)
        resized_img = cv2.resize(img, new_size)
        cv2.imshow('img', resized_img)
        cv2.waitKey(150)

print(len(img_points))
cv2.destroyAllWindows()

if len(img_points) > 0:
    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    save_calibration_to_yaml('../data/calibration_danmu.yaml', mtx, dist)  # 改！换成自己yaml文件想要的路径和名字

    print("ret:", ret)
    print("mtx:\n", mtx)
    print("dist:\n", dist)
    print("rvecs:\n", rvecs)
    print("tvecs:\n", tvecs)

else:
    print("没有检测到角点，无法进行相机标定。")
