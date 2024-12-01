import os
import cv2
import numpy as np
from ultralytics import YOLO
from models.downloader import load_yolo11_model, load_sam_model

door_area = 0.8
spray_width = 0.1
nozzle_drift = 0.08
LKM_cost_per_liter = 62500

coverage_per_liter = 10

def extract_elements(image_rgb, contrast_threshold=0.3, min_area=500):
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
    _, thresholded = cv2.threshold(edges, contrast_threshold * 255, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    return filtered_contours

def calculate_processing_area(contour, spray_width=0.1, nozzle_drift=0.08):
    spray_width_cm = spray_width * 100
    nozzle_drift_cm = nozzle_drift * 100

    contour_length = cv2.arcLength(contour, closed=True)

    processing_area_per_pass = spray_width_cm * (contour_length + nozzle_drift_cm)

    return processing_area_per_pass

def calculate_painting_cost(processing_area):
    required_liters = processing_area / (
                coverage_per_liter * 10000)
    total_cost = required_liters * LKM_cost_per_liter
    return total_cost

def process_image(image_path, output_path, yolo_model, sam_predictor, min_area=500):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    if len(detections) == 0:
        print(f"На изображении {image_path} не найдено частей автомобиля.")
        return

    parts_classes = {'hood': 0, 'door': 1, 'wing': 2}

    overlay = image_rgb.copy()

    total_processing_area = 0
    part_processing_areas = {}

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        class_name = results[0].names[int(class_id)]

        if class_name in parts_classes:
            x_center = (x1 + x2) / 2.0 / image_rgb.shape[1]
            y_center = (y1 + y2) / 2.0 / image_rgb.shape[0]
            width = (x2 - x1) / image_rgb.shape[1]
            height = (y2 - y1) / image_rgb.shape[0]

            color = (0, 255, 0) if class_name == 'hood' else (0, 0, 255) if class_name == 'door' else (255, 0, 0)
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            box = np.array([x1, y1, x2, y2]).astype(int)
            sam_predictor.set_image(image_rgb)
            masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
            best_mask_index = np.argmax(scores)
            best_mask = masks[best_mask_index]

            contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

            for contour in filtered_contours:
                processing_area = calculate_processing_area(contour)
                total_processing_area += processing_area

                if class_name not in part_processing_areas:
                    part_processing_areas[class_name] = []
                part_processing_areas[class_name].append(processing_area)

                cv2.drawContours(overlay, [contour], -1, color, thickness=2)

    output_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr)
    print(f"Сохранено: {output_path}")

    for part, areas in part_processing_areas.items():
        for i, area in enumerate(areas, start=1):
            cost = calculate_painting_cost(area)
            print(f"{part} {i} - {area:.2f} см², Стоимость: {cost:.2f} руб.")

def main():
    input_dir = "./assets/unedited/jpgs"
    output_dir = "./assets/edited"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    yolo_model = YOLO("./train14/weights/best.pt")
    sam_model = load_sam_model("models/sam_vit_h_4b8939.pth")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            try:
                process_image(input_path, output_path, yolo_model, sam_model)
            except Exception as e:
                print(f"Ошибка обработки {filename}: {e}")


if __name__ == "__main__":
    main()