from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")
import ultralytics.utils.globals as globals_value 


def train():
    globals_value.cls_vector_flag = True
    model = YOLO(r"ultralytics/cfg/models/baseline-FSAM.yaml",task='obb')  

    # 训练
    model.train(
        data=r"ultralytics/cfg/datasets/droneVehicle.yaml",
        batch=16,
        epochs=200,
        workers=8,
        device='0',
        optimizer="SGD", 
        lr0=0.02,    
        weight_decay=0.001,
        # 训练策略
        cos_lr=True
    ) 



if __name__ == "__main__":
    train() 