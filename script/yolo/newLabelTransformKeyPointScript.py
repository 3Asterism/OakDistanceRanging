import json
import os
from pathlib import Path


def convert_labelme_to_yolo_pose(json_path, image_width=1280, image_height=800):
    """
    将labelme的JSON标注转换为YOLOv8-pose格式
    专门处理head边界框和top关键点

    Args:
        json_path: JSON文件路径
        image_width: 图片宽度（1280）
        image_height: 图片高度（800）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 分别存储head边界框和top关键点
    head_boxes = []
    top_points = []

    # 处理所有标注
    for shape in data['shapes']:
        if shape['label'] == 'head' and shape['shape_type'] == 'rectangle':
            # 获取矩形框坐标
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            # 确保坐标顺序正确
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            # 计算中心点和宽高并归一化
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            head_boxes.append([x_center, y_center, width, height])

        elif shape['label'] == 'top' and shape['shape_type'] == 'point':
            # 获取关键点坐标并归一化
            x, y = shape['points'][0]
            x_norm = x / image_width
            y_norm = y / image_height
            top_points.append([x_norm, y_norm])

    # 检查数量是否匹配
    if len(head_boxes) != len(top_points):
        print(f"警告：在文件 {json_path} 中")
        print(f"head边界框数量 ({len(head_boxes)}) 与 top关键点数量 ({len(top_points)}) 不匹配！")
        return []

    # 生成YOLO格式的标注
    yolo_annotations = []
    for i in range(len(head_boxes)):
        # format: class_id x_center y_center width height x_point y_point conf
        annotation = [0]  # class_id = 0
        annotation.extend(head_boxes[i])  # 边界框信息
        annotation.extend(top_points[i])  # 关键点坐标
        annotation.append(1.0)  # 关键点置信度
        yolo_annotations.append(annotation)

    return yolo_annotations


def convert_and_save(json_dir, output_dir):
    """
    转换整个文件夹的JSON文件并保存为txt格式
    """
    os.makedirs(output_dir, exist_ok=True)

    for json_file in Path(json_dir).glob('*.json'):
        print(f"处理文件: {json_file.name}")
        try:
            annotations = convert_labelme_to_yolo_pose(json_file)

            if not annotations:
                print(f"警告：{json_file.name} 没有生成有效的标注")
                continue

            # 创建对应的txt文件
            txt_path = Path(output_dir) / f"{json_file.stem}.txt"
            with open(txt_path, 'w') as f:
                for annotation in annotations:
                    # 将数字转换为字符串，保留6位小数
                    line = ' '.join(map(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x), annotation))
                    f.write(f"{line}\n")
            print(f"成功处理: {txt_path} ({len(annotations)} 个标注)")

        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {str(e)}")


def verify_conversion(txt_path):
    """
    验证转换后的标注格式是否正确
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    print(f"\n验证文件：{txt_path}")
    print(f"包含 {len(lines)} 个标注")

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 8:  # class_id + 4(box) + 2(point) + 1(conf) = 8
            print(f"警告：第 {i} 行格式不正确，应该有8个值，实际有 {len(parts)} 个")
            print(f"行内容：{line.strip()}")
        else:
            print(f"标注 {i}: {line.strip()}")


# 使用示例
if __name__ == "__main__":
    # 设置路径
    json_dir = r"C:\dataSet\keypoint\label"  # 你的JSON文件夹路径
    output_dir = r"C:\dataSet\keypoint\testTxt"  # 输出的标签文件夹路径

    # 执行转换
    convert_and_save(json_dir, output_dir)

    # 验证第一个转换后的文件（可选）
    if os.path.exists(output_dir):
        txt_files = list(Path(output_dir).glob('*.txt'))
        if txt_files:
            verify_conversion(txt_files[0])
