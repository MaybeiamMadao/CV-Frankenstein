import os
from ultralytics import YOLO

def main():
    # 避免多库冲突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    ROOT_DIR = 'D:/cvproject/projct/my/chess/data/'
    yaml_path = os.path.join(ROOT_DIR , 'data.yaml')
    model = YOLO('yolov8s.pt')

    results = model.train(
        data=yaml_path,
        epochs=20,
        batch=8,
        imgsz=640,
        lr0=1e-6,
        plots=True,
        verbose=True,
        workers=2,
        cache=True
    )

    print("训练完成。日志保存在:", results.save_dir)

if __name__ == "__main__":
    main()
