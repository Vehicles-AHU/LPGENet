# LPGENet
### Installation

```bash
# Install PyTorch (choose the appropriate command based on your CUDA version)
pip install torch torchvision

# Install other dependencies
pip install numpy opencv-python pillow matplotlib tqdm pyyaml scikit-learn
pip install git+https://github.com/openai/CLIP.git

# Install project dependencies
cd ultralytics_droneVehicle
pip install -e .
```


### Dataset Preparation

#### DroneVehicle Dataset
```
/data/datasets/DroneVehicle/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
Use the tools in `DOTA_devkit/` for data preprocessing and format conversion.



### Model Training

```python
# Edit train.py to configure training parameters
python train.py
```

Training Configuration Example:
- Batch size: 16
- Epochs: 200
- Optimizer: SGD
- Initial learning rate: 0.02
- Learning rate decay strategy: Cosine Annealing
- Early stopping patience: 10 epochs

Run in background:
```bash
python train.py
```

