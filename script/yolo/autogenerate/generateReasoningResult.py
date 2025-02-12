import os
import json
import cv2
from ultralytics import YOLO


def generate_labelme_annotation(image_path, model):
    """
    对单张图像进行 YOLO 推理，并生成符合 labelme 格式的标注数据（字典格式）。

    该关键点模型每个检测结果同时输出一个边界框和对应的关键点，
    本函数为每个检测结果生成两个 shape：
      - 矩形 shape（label 为 "head"）：由边界框的左上角和右下角组成
      - 点 shape（label 为 "top"）：取关键点中的第一个关键点（只保留 x, y 坐标）

    参数:
        image_path: str
            待检测图像的完整路径。
        model: YOLO
            已加载的 YOLO 关键点检测模型对象。

    返回:
        annotation: dict
            按 labelme 格式组织的标注数据字典；若图像读取失败则返回 None。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 获取图像高度和宽度
    height, width = image.shape[:2]

    # 使用模型进行推理
    results = model(image)

    shapes = []  # 存储所有生成的 shape

    # 遍历推理结果（通常每张图像返回一个结果对象）
    for r in results:
        # 检查是否同时存在边界框和关键点数据
        if r.boxes is None or r.keypoints is None:
            continue

        # 将检测框数据转换为 numpy 数组
        try:
            boxes_data = r.boxes.data.cpu().numpy()
        except AttributeError:
            boxes_data = r.boxes.data

        # 将关键点数据转换为 numpy 数组
        try:
            keypoints_data = r.keypoints.data.cpu().numpy()
        except AttributeError:
            keypoints_data = r.keypoints.data

        # 遍历每个检测结果（假设检测框和关键点数量一致）
        num_detections = len(boxes_data)
        for i in range(num_detections):
            # 获取当前检测的边界框坐标 (x1, y1, x2, y2)
            box = boxes_data[i]
            x1, y1, x2, y2 = box[:4]

            # 构造矩形 shape，标签设为 "head"
            rectangle_shape = {
                "label": "head",
                "points": [
                    [float(x1), float(y1)],
                    [float(x2), float(y2)]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            shapes.append(rectangle_shape)

            # 获取当前检测对应的关键点数据
            kp = keypoints_data[i]
            # 判断关键点数据是否为二维数组（即多个关键点的情况），取第一个关键点
            if hasattr(kp, 'ndim') and kp.ndim == 2:
                keypoint = kp[0]
            else:
                keypoint = kp
            # 取前两个值作为关键点的 x, y 坐标
            kp_x, kp_y = keypoint[:2]

            # 构造点 shape，标签设为 "top"
            point_shape = {
                "label": "top",
                "points": [
                    [float(kp_x), float(kp_y)]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
                "mask": None
            }
            shapes.append(point_shape)

    # 构造 JSON 中 imagePath 字段：
    # 由于 JSON 文件保存到 label 文件夹，而图片在 pic 文件夹，
    # 因此 imagePath 使用相对于 label 文件夹的路径格式，如 "..\\pic\\xxx.jpg"
    image_filename = os.path.basename(image_path)
    relative_image_path = f"..\\pic\\{image_filename}"

    # 构造最终的标注数据字典（符合 labelme 的 JSON 格式）
    annotation = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": relative_image_path,
        "imageData": None,  # 必须字段，设为 None 序列化后为 null
        "imageHeight": height,
        "imageWidth": width
    }
    return annotation


def main():
    # 定义图片所在文件夹和 JSON 标注文件保存的文件夹
    pic_folder = r"C:\dataSet\autoTest\pic"  # 推理图片所在的文件夹
    label_folder = r"C:\dataSet\autoTest\label"  # 保存生成的 JSON 文件的文件夹
    os.makedirs(label_folder, exist_ok=True)

    # 加载训练好的关键点模型（请修改模型路径为实际路径）
    model_path = r"C:\dataSet\weights\pose\train32\weights\best.pt"
    model = YOLO(model_path)

    # 遍历图片文件夹中的所有图像文件（支持 .jpg, .jpeg, .png 格式）
    for filename in os.listdir(pic_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(pic_folder, filename)
            print(f"正在处理图像: {image_path}")

            # 调用生成标注数据的函数
            annotation = generate_labelme_annotation(image_path, model)
            if annotation is None:
                continue

            # 生成对应的 JSON 文件名（与图片名同名，只是扩展名变为 .json）
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(label_folder, json_filename)

            # 将标注数据写入 JSON 文件，格式化输出（缩进 4 格，确保中文正常显示）
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=4, ensure_ascii=False)
            print(f"标注保存到: {json_path}")


if __name__ == "__main__":
    main()
