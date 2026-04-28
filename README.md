# TROV Segformer Traversability

A ROS2 node that runs real-time semantic segmentation on the robot's RealSense camera feed and classifies every pixel into one of three traversability categories — safe, risky, or blocked. Built on SegFormer-B5 fine-tuned on the ADE20K dataset, running in FP16 on the Jetson AGX Orin GPU.

This is part of the TROV autonomous ground vehicle system. See the main [trov repository](https://github.com/adhith-edgeforce/trov) for the full system.

---

## What it does

Every frame from the RealSense camera goes through the following pipeline:

```
/camera/camera/color/image_raw  (RealSense RGB)
    ↓  CLAHE contrast enhancement (LAB colour space)
    ↓  Resize to 512×512
    ↓  SegFormer-B5 inference  (FP16, CUDA)
    ↓  Bilinear upsample back to original resolution
    ↓  Pixel-wise class → traversability lookup
    ↓
/fusion_segmentation/traversability   (green/yellow/red map)
/fusion_segmentation/semantic         (full colour semantic map)
```

---

## Traversability categories

Every ADE20K class is mapped to one of three categories:

| Category | Colour | Examples |
|---|---|---|
| Safe | Green `(0, 255, 0)` | Floor, road, path, ground, carpet, grass, tile |
| Risky | Yellow `(0, 255, 255)` | Stairs, ramps, furniture, vegetation |
| Blocked | Red `(0, 0, 255)` | Walls, doors, people, vehicles, water, sky |

Everything that isn't explicitly safe or risky defaults to blocked. The top 1/6th of the frame is additionally forced to blocked if it isn't already sky-coloured — this prevents ceiling and overhead structure from being misclassified as traversable.

The semantic output uses distinct colours per class (floor = light grey, grass = bright green, wall = dark grey, person = dark red, water = cyan, etc.) for debugging and visualisation.

---

## Technical details

**Model** — SegFormer-B5 from HuggingFace Transformers, fine-tuned on ADE20K (150 semantic classes). Loaded from a local directory at `/home/nvidia/segimage/models/segformer-b5-ade`. The model runs in FP16 with `torch.amp.autocast` on the Jetson GPU, which roughly halves memory usage and speeds up inference compared to FP32.

**Preprocessing** — each frame goes through CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB colour space before inference. This enhances local contrast and makes the model significantly more robust in low-light warehouse environments where a flat histogram would cause misclassification.

**Warmup** — on startup, a dummy frame is passed through the full pipeline to warm up CUDA kernels before any real frames arrive. Without this, the first real frame takes several seconds longer than subsequent ones as CUDA JIT-compiles the kernels.

**Timing** — preprocessing, inference, and postprocessing are each timed separately. Every 10 frames, the node logs the average time for each stage along with overall FPS. On shutdown, final statistics across all processed frames are printed to the terminal.

**Publishing** — both output images are published as `sensor_msgs/Image` in `bgr8` encoding using `cv_bridge`. The header timestamp is copied from the input frame so downstream consumers can correlate outputs with the original camera timestamp.

---

## ROS2 interface

| Direction | Topic | Type | Description |
|---|---|---|---|
| Subscribed | `/camera/camera/color/image_raw` | `sensor_msgs/Image` | RealSense RGB input |
| Published | `/fusion_segmentation/traversability` | `sensor_msgs/Image` | Three-colour traversability map |
| Published | `/fusion_segmentation/semantic` | `sensor_msgs/Image` | Full semantic colour map |

---

## Status

The node is running and producing correct output. Integration with the Nav2 navigation stack is the next planned step — the traversability map will be consumed by a custom costmap layer so the planner can avoid risky and blocked regions detected by the camera, in addition to the existing LiDAR obstacle layer. This is particularly useful for detecting low obstacles, transparent surfaces, and terrain changes that the LiDAR misses.

---

## Setup

### 1. Install the model

Download the SegFormer-B5 ADE20K model from HuggingFace and place it at:

```
/home/nvidia/segimage/models/segformer-b5-ade/
```

The directory must contain the standard HuggingFace model files — `config.json`, `pytorch_model.bin` (or safetensors), and `preprocessor_config.json`.

### 2. Create the virtual environment

```bash
python3 -m venv /data/segformer_env
source /data/segformer_env/bin/activate
pip install -r requirements.txt
deactivate
```

Note that `torch` and `numpy` come pre-installed with JetPack on the Jetson and do not need to be pip-installed. `rclpy` and `cv_bridge` come from the ROS2 installation.

### 3. Build the ROS2 package

```bash
cd /data/trov_ws
colcon build --packages-select segformer_traversability_autoware
source install/setup.bash
```

### 4. Run

```bash
./launch_segformer.sh
```

The launch script (`launch_segformer.sh` in the main trov repo) handles sourcing the virtual environment and the ROS2 workspace before starting the node.

Or run manually:

```bash
source /data/segformer_env/bin/activate
source /data/trov_ws/install/setup.bash
export PYTHONPATH=/data/segformer_env/lib/python3.10/site-packages:$PYTHONPATH
ros2 run segformer_traversability_autoware segformer_node
```

### 5. View output in RViz

Add two Image displays in RViz:

- `/fusion_segmentation/traversability` — green/yellow/red traversability map
- `/fusion_segmentation/semantic` — full semantic colour map

---

## Dependencies

| Package | Version | Source |
|---|---|---|
| `transformers` | 4.44.2 | pip |
| `accelerate` | 1.13.0 | pip |
| `opencv-python` | 4.13.0.92 | pip |
| `torchvision` | 0.20.0 | pip |
| `torch` | 2.5.0 | JetPack (pre-installed) |
| `numpy` | — | JetPack (pre-installed) |
| `rclpy` | — | ROS2 Humble (system) |
| `cv_bridge` | — | ROS2 Humble (system) |
