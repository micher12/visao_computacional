from ultralytics import YOLO

model = YOLO("best12n_v4.pt")


if __name__ == '__main__':
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        name='yolo12n_v5',
        device="cuda",
    )