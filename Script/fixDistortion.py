import cv2
import numpy as np
import glob
import os
import yaml


def load_camera_params_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
    dist_coeffs = np.array(calib_data['dist_coeff']['data'])

    return camera_matrix, dist_coeffs


def cam_calib_correct_img(distort_img_dir, crct_img_dir, cameraMatrix, distCoeffs):
    imgs = glob.glob(os.path.join(distort_img_dir, "*.png"))
    imgs.extend(glob.glob(os.path.join(distort_img_dir, "*.png")))
    imgs.extend(glob.glob(os.path.join(distort_img_dir, "*.bmp")))

    for img_ in imgs:
        print("已读取待校正图像：", img_)
        img = cv2.imread(img_)
        (h1, w1) = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w1, h1), 1, (w1, h1))

        dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        resized_dst = cv2.resize(dst, (w1, h1))

        imgname = os.path.basename(img_).split('.')[0]
        rlt_path = os.path.join(crct_img_dir, imgname + "_crct.png")
        cv2.imwrite(rlt_path, resized_dst)

        print("已保存校正图像：", rlt_path)


if __name__ == "__main__":
    distort_img_dir = 'pic'  # 待校正图像的路径
    crct_img_dir = 'picFix'  # 保存校正图像的路径
    yaml_file = 'calibration_danmu_old.yaml'  # YAML文件路径

    cameraMatrix, distCoeffs = load_camera_params_from_yaml(yaml_file)
    cam_calib_correct_img(distort_img_dir, crct_img_dir, cameraMatrix, distCoeffs)

