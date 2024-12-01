import json
import os

def labelme_to_yolo(labelme_json_dir, yolo_txt_dir, class_map):
    if not os.path.exists(yolo_txt_dir):
        os.makedirs(yolo_txt_dir)

    for json_file in os.listdir(labelme_json_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(labelme_json_dir, json_file), 'r') as f:
                data = json.load(f)

            img_width = data['imageWidth']
            img_height = data['imageHeight']

            yolo_annotations = []
            for shape in data['shapes']:
                label = shape['label']
                if label not in class_map:
                    continue

                points = shape['points']
                x_min, y_min = min(p[0] for p in points), min(p[1] for p in points)
                x_max, y_max = max(p[0] for p in points), max(p[1] for p in points)

                x_center = (x_min + x_max) / 2.0 / img_width
                y_center = (y_min + y_max) / 2.0 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                class_id = class_map[label]
                yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

            yolo_txt_file = os.path.join(yolo_txt_dir, json_file.replace('.json', '.txt'))
            with open(yolo_txt_file, 'w') as f:
                f.write("\n".join(yolo_annotations))

labelme_json_dir = "assets/unedited/annotations"
yolo_txt_dir = "assets/yolo_annotations"
class_map = {"hood": 0, "door": 1, "wing": 2}

labelme_to_yolo(labelme_json_dir, yolo_txt_dir, class_map)
