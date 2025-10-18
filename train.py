from ultralytics import YOLO

model = YOLO("runs/detect/yolo11n_new_hat/weights/best.pt")


if __name__ == '__main__':
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        name='yolo11n_new_hatv2',
        device="cuda",
    )