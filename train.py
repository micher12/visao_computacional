from ultralytics import YOLO

model = YOLO("runs/detect/yolov12n_engenharia_civil3/weights/best.pt")

if __name__ == '__main__':
    results = model.train(
        data='datasets/data.yaml',
        epochs=100,
        imgsz=640,
        name='yolov12n_engenharia_civil4',
        device="cuda",
        mosaic=1.0
    )
