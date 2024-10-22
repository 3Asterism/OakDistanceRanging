import os
import cv2

"""将彩图转换为灰度图"""

# 输入和输出文件夹路径
input_folder = input("Enter the path to the folder containing RGB pics: ").replace('\\', '/')
output_folder = input("Enter the path to the folder to save gray pics: ").replace('\\', '/')

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 只处理.jpg文件
    if filename.endswith('.jpg') or filename.endswith('.JPG'):
        # 构建完整的文件路径
        img_path = os.path.join(input_folder, filename)

        # 读取彩色图像
        img = cv2.imread(img_path)

        # 将图像转换为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 构建输出文件的完整路径
        output_path = os.path.join(output_folder, filename)

        # 保存灰度图像
        cv2.imwrite(output_path, gray_img)

print('图像转换完成！')
