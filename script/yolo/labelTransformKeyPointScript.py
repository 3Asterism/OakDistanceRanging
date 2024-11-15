import json
import os
import math


def calculate_distance(x1, y1, x2, y2):
    """计算两点之间的欧氏距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def convert_labelme_to_yolo(labelme_dir, output_dir, image_width, image_height):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历目录中的所有 JSON 文件
    for filename in os.listdir(labelme_dir):
        if filename.endswith('.json'):
            labelme_json_path = os.path.join(labelme_dir, filename)

            # 加载 Labelme 的 JSON 文件
            with open(labelme_json_path, 'r') as f:
                data = json.load(f)

            # 创建输出 TXT 文件
            output_txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
            with open(output_txt_path, 'w') as txt_file:
                heads = []
                tops = []

                # 遍历所有标注对象并分离 head 和 top
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']

                    if label == 'head':
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # 转换为 YOLO 格式
                        x_center = (x_min + x_max) / 2 / image_width
                        y_center = (y_min + y_max) / 2 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height

                        heads.append((x_center, y_center, width, height))

                    elif label == 'top':
                        for point in points:
                            top_x, top_y = point
                            x_center = top_x / image_width
                            y_center = top_y / image_height
                            tops.append((x_center, y_center))

                # 匹配每个 head 与最近的 top
                matched_tops = set()
                for head_bbox in heads:
                    head_x, head_y, head_w, head_h = head_bbox
                    min_distance = float('inf')
                    closest_top = None

                    for top_point in tops:
                        top_x, top_y = top_point
                        distance = calculate_distance(head_x * image_width, head_y * image_height,
                                                      top_x * image_width, top_y * image_height)

                        if distance < min_distance and top_point not in matched_tops:
                            min_distance = distance
                            closest_top = top_point

                    if closest_top:
                        matched_tops.add(closest_top)
                        visibility = 1  # 假设关键点可见
                        keypoints = [closest_top[0], closest_top[1]]

                        # 输出 YOLO 格式数据
                        class_id = 0  # 假设类别 ID 为 0
                        yolo_annotation = (
                                f"{class_id} {head_x} {head_y} {head_w} {head_h} "
                                + " ".join(map(str, keypoints)) + "\n"
                        )
                        txt_file.write(yolo_annotation)


# 使用示例
labelme_dir = 'C:/dataSet/keypoint/valLabel'  # Labelme JSON 文件所在目录
output_dir = 'C:/dataSet/keypoint/valTxt'  # 输出目录
image_width, image_height = 1280, 800  # 图像宽高

convert_labelme_to_yolo(labelme_dir, output_dir, image_width, image_height)
