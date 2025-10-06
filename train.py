from ultralytics import YOLO

model = YOLO("runs/detect/yolo12n_finalv2/weights/best.pt")


if __name__ == '__main__':
    results = model.train(
        data='datasets/data.yaml',
        epochs=100,
        imgsz=640,
        name='yolo12n_finalv3',
        device="cuda",
        mosaic=1.0
    )
