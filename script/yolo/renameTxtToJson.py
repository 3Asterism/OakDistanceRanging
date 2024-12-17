import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = r"C:\dataSet\box\valTxt"  # 源文件夹路径，存放 .txt 文件
target_folder = r"C:\dataSet\box\valLabel"  # 目标文件夹路径，存放 .json 文件

# 如果目标文件夹不存在，则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)  # 创建目标文件夹

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 检查文件是否以 .txt 结尾
    if filename.endswith('.txt'):
        # 构建源文件的完整路径
        source_path = os.path.join(source_folder, filename)
        # 替换文件后缀为 .json，保留原文件名
        new_filename = filename.replace('.txt', '.json')
        # 构建目标文件的完整路径
        target_path = os.path.join(target_folder, new_filename)
        # 复制文件到目标文件夹，并将后缀改为 .json
        shutil.copyfile(source_path, target_path)  # 完成复制和重命名

print("所有 .txt 文件已成功转换为 .json 并移动到目标文件夹。")
