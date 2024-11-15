import os
import shutil


def clean_unpaired_files(pic_path, txt_path):
    """
    清理txt文件夹中没有对应jpg文件的txt文件

    Args:
        pic_path: 存放jpg文件的文件夹路径
        txt_path: 存放txt文件的文件夹路径
    """
    # 获取所有jpg文件的文件名（不含扩展名）
    jpg_files = set()
    for file in os.listdir(pic_path):
        if file.endswith('.jpg'):
            jpg_files.add(os.path.splitext(file)[0])

    # 检查所有txt文件
    removed_count = 0
    for file in os.listdir(txt_path):
        if file.endswith('.txt'):
            file_name = os.path.splitext(file)[0]
            # 如果txt文件没有对应的jpg文件
            if file_name not in jpg_files:
                file_path = os.path.join(txt_path, file)
                try:
                    # 删除文件
                    os.remove(file_path)
                    removed_count += 1
                    print(f"已删除: {file}")
                except Exception as e:
                    print(f"删除 {file} 时出错: {str(e)}")

    print(f"\n清理完成！共删除了 {removed_count} 个无对应jpg文件的txt文件")


# 使用示例
if __name__ == "__main__":
    pic_folder = r"C:\dataSet\keypoint\pic"
    txt_folder = r"C:\dataSet\keypoint\label"

    # 确保路径存在
    if not os.path.exists(pic_folder):
        print(f"错误：图片文件夹路径不存在: {pic_folder}")
    elif not os.path.exists(txt_folder):
        print(f"错误：文本文件夹路径不存在: {txt_folder}")
    else:
        # 执行清理
        clean_unpaired_files(pic_folder, txt_folder)