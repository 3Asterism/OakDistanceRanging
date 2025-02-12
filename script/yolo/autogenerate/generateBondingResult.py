import os
import json
import cv2
from ultralytics import YOLO


def generate_labelme_annotation(image_path, model):
    """
    对单张图像进行检测，并生成符合 labelme 格式的标注数据（仅包含 head 字段）。
    每个检测结果生成一个矩形 shape，标签设为 "head"，使用检测到的边界框坐标。

    参数:
        image_path: str
            待检测图像的完整路径。
        model: YOLO
            已加载的 YOLO 模型对象。

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

    # 使用模型对图像进行检测
    results = model(image)

    shapes = []  # 存储所有生成的 shape（仅 head）

    # 遍历推理结果（通常每张图像返回一个结果对象）
    for r in results:
        if r.boxes is None:
            continue

        # 尝试将检测框数据转换为 numpy 数组（兼容 torch tensor 或直接数据）
        try:
            boxes_data = r.boxes.data.cpu().numpy()
        except AttributeError:
            boxes_data = r.boxes.data

        # 遍历每个检测框，格式一般为 [x1, y1, x2, y2, conf, ...]
        for box in boxes_data:
            x1, y1, x2, y2 = box[:4]
            # 构造矩形 shape，标签设为 "head"
            shape = {
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
            shapes.append(shape)

    # 构造 JSON 中 imagePath 字段
    # 假设 JSON 文件保存到 label 文件夹，而图片存放在 pic 文件夹，
    # imagePath 使用相对于 label 文件夹的路径格式，如 "..\\pic\\xxx.jpg"
    image_filename = os.path.basename(image_path)
    relative_image_path = f"..\\pic\\{image_filename}"

    # 构造最终的标注数据字典（符合 labelme JSON 格式）
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
    # 定义图片所在文件夹和生成 JSON 标注文件保存的文件夹
    pic_folder = r"C:\dataSet\autoTest\pic"  # 待检测图像所在的文件夹
    label_folder = r"C:\dataSet\autoTest\label"  # 保存生成 JSON 文件的文件夹
    os.makedirs(label_folder, exist_ok=True)

    # 加载训练好的模型（请修改模型路径为实际路径）
    model_path = r"C:\dataSet\weights\pose\train32\weights\best.pt"
    model = YOLO(model_path)

    # 遍历图片文件夹中的所有图像文件（支持 .jpg, .jpeg, .png 格式）
    for filename in os.listdir(pic_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(pic_folder, filename)
            print(f"正在处理图像: {image_path}")

            # 生成标注数据（仅包含 head 检测结果）
            annotation = generate_labelme_annotation(image_path, model)
            if annotation is None:
                continue

            # 生成对应的 JSON 文件名（与图片同名，只是扩展名变为 .json）
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(label_folder, json_filename)

            # 将标注数据写入 JSON 文件，格式化输出（缩进 4 格，确保中文正常显示）
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=4, ensure_ascii=False)
            print(f"标注保存到: {json_path}")


if __name__ == "__main__":
    main()
