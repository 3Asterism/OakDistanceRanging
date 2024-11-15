import os
import json

"""转换模型 json->txt"""

# 定义路径
json_folder = input("Enter the path to the folder containing JSON files: ").replace('\\', '/')
txt_folder = input("Enter the path to the folder to save TXT files: ").replace('\\', '/')

# 创建输出文件夹（如果不存在）
os.makedirs(txt_folder, exist_ok=True)

# 标签映射字典，将标签映射为整数
# label_map = {
#     "empty": 0,
#     "full": 1,
#     # 添加其他标签映射
# }

label_map = {
    "key": 0,
}

# 获取所有JSON文件
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# 遍历所有JSON文件
for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像宽度和高度
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 初始化一个空的TXT内容
    txt_content = []

    # 遍历所有标注的对象
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        # 使用标签映射字典
        yolo_label = label_map.get(label, -1)
        if yolo_label == -1:
            continue  # 跳过未定义的标签

        # YOLO格式需要计算中心点和宽高
        x_min = min(point[0] for point in points)
        x_max = max(point[0] for point in points)
        y_min = min(point[1] for point in points)
        y_max = max(point[1] for point in points)

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # 将标签和坐标格式化为YOLO需要的格式
        txt_content.append(f"{yolo_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 将TXT内容写入文件
    txt_file = os.path.join(txt_folder, json_file.replace('.json', '.txt'))
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_content))

print("转换完成！")
