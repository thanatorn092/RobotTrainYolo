# RobotTrainYolo

**- Robotics and AI**: a differential-drive robot in CoppeliaSim learns to tell a **Cube** from a **Tree** with a YOLO detector trained on my own simulation images, then builds an object map using wheel odometry while rotating on the spot.

```
capture images → label (Label Studio) → split train/val → train YOLO → evaluate → odometry + detector = object map
```

## Tools

- CoppeliaSim Edu 4.10 (controlled from Python over the ZMQ Remote API, port 23000)
- Python 3.13, Ultralytics YOLO11n, OpenCV, NumPy, Matplotlib
- Label Studio (bounding-box annotation)

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Robot controller: rotates once, saves camera frames, runs YOLO, integrates wheel odometry, writes the object map |
| `zmq_demo.py` | Minimal example of controlling CoppeliaSim from Python |
| `gather_images.py` | Collects frames from all capture runs into one folder |
| `prepare_dataset.py` | Splits a Label Studio YOLO export into train/val and writes `dataset.yaml` |
| `train.py` / `evaluate.py` | Train and evaluate the detector |
| `scenes/` | CoppeliaSim scenes (`Demo_CV.ttt` original, `my_scene.ttt` used for capturing) |
| `dataset/` | Labelled dataset (Cube = 0, Tree = 1) |
| `runs/` | Training runs (`best.pt`, curves) and evaluation results |
| `output/` | Final object map: `map.png`, `objects.json`, `odometry.csv` |

Not included (too large or re-creatable): virtual environments, raw captures, Label Studio export, `yolo11n.pt` (Ultralytics downloads it automatically).

## Setup

```bash
python3 -m venv yolo
source yolo/bin/activate            # Windows: .\yolo\Scripts\Activate.ps1
pip install -r requirements.txt
```

Label Studio runs in its own environment:

```bash
python3 -m venv ls
source ls/bin/activate
pip install -U label-studio
label-studio                        # http://localhost:8080
```

> `dataset/dataset.yaml` contains an absolute `path`. Change it to the `dataset/` folder on your machine before training.

## How to run

Open CoppeliaSim with the scene and leave the simulation **stopped** (`main.py` starts and stops it).

```bash
# 1. Capture (set save_frames=True and output_dir="captures/runN" in main.py, once per scene layout)
python main.py --weights yolo11n.pt
python gather_images.py --step 4

# 2. Label in Label Studio (classes: Cube, Tree), export as "YOLO with Images" into export/

# 3. Split by scene: runs 1–4 for training, run 5 for validation
python prepare_dataset.py --source export --train-pattern "*run[1-4]_*" --val-pattern "*run5_*"

# 4. Train with and without augmentation
python train.py --data dataset/dataset.yaml --epochs 100 --augmentation none --name objects_no_aug --device cpu --imgsz 320
python train.py --data dataset/dataset.yaml --epochs 100 --augmentation mild --name objects_augmented --device cpu --imgsz 320

# 5. Evaluate
python evaluate.py --weights runs/objects_augmented/weights/best.pt --data dataset/dataset.yaml --split val --device cpu --preview --imgsz 320

# 6. Map objects (scene_path = Demo_CV.ttt, output_dir = "output")
python main.py --weights runs/objects_augmented/weights/best.pt
```

## Dataset

117 labelled images from 5 capture runs with different robot positions and object layouts (more objects, near/far objects, Trees only, partial occlusion). The Bowl is kept in the scene as an unlabelled distractor. The split is by scene to avoid near-duplicate frames in both sets: **94 train** (runs 1–4) and **23 validation** (run 5).

## Results

### Detector (validation set)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| No augmentation | 0.659 | 0.625 | 0.491 | 0.388 |
| **Mild augmentation** | **1.000** | 0.605 | **0.707** | **0.526** |

The augmented model is better. Without augmentation the model overfits: training loss falls to almost zero while validation mAP peaks in the first epochs. Augmentation forces the network to learn the objects' real features and improves results on the unseen run.

**Cube is the weaker class** (mAP50 0.34 vs. 0.64 for Tree, no-augmentation model). The validation set contains only 3 Cubes, one of which was mislabelled.

Example errors:
- `run5_rgb_000060`: both objects missed. The Cube was mislabelled as Tree, and it hides the Tree's trunk so only the canopy is visible.
- `run5_rgb_000080`: the Tree is cut off by the image edge. The model found it, but its box was too tall (IoU < 0.5), so it counts as a miss and a false positive.

### Object map (original scene)

Robot start (world): (0.100, −0.700). Map positions are in the robot's start frame.

| Detected | Map (x, y) | True (x, y) | Error |
|---|---|---|---|
| Tree | (0.350, −1.029) | (0.374, −1.086) | 0.06 m |
| Cube | (0.592, −0.359) | (0.600, −0.525) | 0.17 m |

Both real objects were found (2/2), with no false negatives and no false positives (the Bowl was ignored). Mapped positions lie slightly closer to the robot than the true centres because the depth camera measures each object's front surface, not its centre.

## Limitations

- Small validation set (15 objects, 3 Cubes), so per-class scores are noisy.
- Wheel odometry drifts (wheel slip); rotating on the spot keeps the drift small here.
- Objects cut off by the image edge or heavily occluded are detected less reliably.
