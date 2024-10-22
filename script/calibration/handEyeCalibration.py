import json
import cv2
import numpy as np

# 从文件读取位姿数据
with open('poses_data.json', 'r') as f:
    data = json.load(f)
    end_effector_poses = data['end_effector_poses']
    camera_poses = data['camera_poses']


# 将机械臂末端位姿转换为旋转矩阵和平移向量
def convert_arm_poses_to_rt(poses):
    rotations = []
    translations = []
    for pose in poses:
        x, y, z, rHead = pose
        rot_matrix = cv2.Rodrigues(np.array([0, 0, rHead]))[0]
        translations.append(np.array([x, y, z]).reshape(3, 1))
        rotations.append(rot_matrix)
    return rotations, translations


# 将相机位姿转换为旋转矩阵和平移向量
def convert_camera_poses_to_rt(poses):
    rotations = []
    translations = []
    for rvec, tvec in poses:
        rot_matrix = cv2.Rodrigues(rvec)[0]
        translations.append(tvec)
        rotations.append(rot_matrix)
    return rotations, translations


# 获取机械臂末端和相机的旋转和平移矩阵
arm_rotations, arm_translations = convert_arm_poses_to_rt(end_effector_poses)
cam_rotations, cam_translations = convert_camera_poses_to_rt(camera_poses)

# 使用Tsai-Lenz手眼标定算法
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    arm_rotations, arm_translations, cam_rotations, cam_translations, method=cv2.CALIB_HAND_EYE_TSAI
)

# 打印结果
print("相机相对于机械臂末端的旋转矩阵:")
print(R_cam2gripper)
print("相机相对于机械臂末端的平移向量:")
print(t_cam2gripper)
