import os
import json
import shutil

"""转换关键点模型 json->txt"""

# 框的类别
bbox_class = {
    'sjb_rect': 0
}

# 关键点的类别
keypoint_class = ['angle_30', 'angle_60', 'angle_90']  # 这里类别放的顺序对应关键点类别的标签 0，1，2


def process_single_json(labelme_path, save_folder):
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']  # 图像宽度
    img_height = labelme['imageHeight']  # 图像高度

    # 生成 YOLO 格式的 txt 文件
    suffix = labelme_path.split('.')[-2]
    yolo_txt_path = suffix + '.txt'

    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
        for each_ann in labelme['shapes']:  # 遍历每个标注
            if each_ann['shape_type'] == 'rectangle':  # 每个框，在 txt 里写一行
                yolo_str = ''
                # 框的信息
                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                bbox_width_norm = bbox_width / img_width
                bbox_height_norm = bbox_height / img_height
                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm,
                                                                  bbox_width_norm, bbox_height_norm)

                # 找到该框中所有关键点，存在字典 bbox_keypoints_dict 中
                bbox_keypoints_dict = {}
                for ann in labelme['shapes']:  # 遍历所有标注
                    if ann['shape_type'] == 'point':  # 筛选出关键点标注
                        x = int(ann['points'][0][0])
                        y = int(ann['points'][0][1])
                        label = ann['label']
                        if (x > bbox_top_left_x) & (x < bbox_bottom_right_x) & (y < bbox_bottom_right_y) & (
                                y > bbox_top_left_y):  # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y]

                # 把关键点按顺序排好
                for each_class in keypoint_class:  # 遍历每一类关键点
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm,
                                                               2)  # 2-可见不遮挡 1-遮挡 0-没有点
                    else:  # 不存在的点，一律为0
                        yolo_str += '0 0 0 '
                f.write(yolo_str + '\n')

    shutil.move(yolo_txt_path, save_folder)
    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))


def main():
    Dataset_root = input("请输入数据集根目录路径: ")
    Output_root = input("请输入输出目录路径: ")

    if not os.path.exists(Dataset_root):
        print("输入的目录不存在")
        return

    labels_dir = os.path.join(Output_root, 'labels')
    train_dir = os.path.join(labels_dir, 'train')
    val_dir = os.path.join(labels_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_json_dir = os.path.join(Dataset_root, 'label_json', 'train')
    val_json_dir = os.path.join(Dataset_root, 'label_json', 'val')

    for json_dir, save_folder in [(train_json_dir, train_dir), (val_json_dir, val_dir)]:
        if not os.path.exists(json_dir):
            print(f"{json_dir} 不存在")
            continue

        for labelme_path in os.listdir(json_dir):
            if labelme_path.endswith('.json'):
                labelme_full_path = os.path.join(json_dir, labelme_path)
                try:
                    process_single_json(labelme_full_path, save_folder=save_folder)
                except Exception as e:
                    print('******有误******', labelme_path, e)


if __name__ == "__main__":
    main()
