from ultralytics import YOLO

def main():
    model = YOLO('yolo11m.pt')
    yolo_trained = model.train(
        data = 'data.yaml',
        imgsz = 640,
        epochs = 100,
        patience = 10,
        batch = 32,
        augment = True
    )
    

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()