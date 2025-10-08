from ultralytics import YOLO

model = YOLO("best.pt")


if __name__ == '__main__':
    results = model.train(
        data="construction-ppe.yaml",
        epochs=100,
        imgsz=640,
        name='yolo11n_detection_v2',
        device="cuda",
    )


