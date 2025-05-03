from ultralytics import YOLO
import cv2
import pytesseract
import os
import csv
from datetime import datetime

def detect_number_plate(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']
   

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    model = YOLO(r"best.pt")

    recognized_numbers = set()
    output_path = ""

    output_dir = "static/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Get original filename without extension and with extension
    base_name = os.path.basename(file_path)
    file_stem, file_ext = os.path.splitext(base_name)

    if is_video:
        cap = cv2.VideoCapture(file_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # Save output video with same name in static/processed
        out_path = os.path.join(output_dir, f"{file_stem}_processed{file_ext}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("[ERROR] Failed to open VideoWriter!")
            return "", "Error: Unable to write video."

        frame_id = 0
        frame_skip = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            results = model(frame, imgsz=512)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                text = pytesseract.image_to_string(
                    gray,
                    config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip().replace('\n', '').replace(' ', '')

                if text and len(text) >= 6:
                    recognized_numbers.add(text)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        output_path = out_path

    else:
        img = cv2.imread(file_path)
        results = model(img, imgsz=512)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            text = pytesseract.image_to_string(
                gray,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip().replace('\n', '').replace(' ', '')

            if text and len(text) >= 6:
                recognized_numbers.add(text)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        #Save image using the original name
        output_path = os.path.join(output_dir, f"{file_stem}_processed{file_ext}")
        cv2.imwrite(output_path, img)

    #Save CSV with timestamp (unchanged)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"recognized_plates_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Recognized Number Plates"])
        for plate in recognized_numbers:
            writer.writerow([plate])

    return output_path.replace("static", "/static"), ", ".join(recognized_numbers) if recognized_numbers else "Not Detected"
