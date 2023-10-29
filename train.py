from ultralytics import YOLO


model = YOLO("yolov8m.pt")

results = model.train(data="config.yaml", epochs=200, batch=10)