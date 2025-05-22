# Eco-Friendly Route Optimization Using Satellite Images 🌱

This project uses deep learning models like U-Net and ResNet to identify environmentally sensitive zones (e.g., forests, water bodies) from satellite images and calculates eco-smart transportation routes.

## Features

- U-Net for land segmentation
- ResNet for road classification
- Graph-based pathfinding (A*, Dijkstra)
- Environmental constraint-aware routing

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eco-smart-routing.git
cd eco-smart-routing
pip install -r requirements.txt
```

## Usage

- Train segmentation model:
  ```bash
  python scripts/train.py
  ```

- Run route optimization:
  ```bash
  python scripts/infer_route.py --image data/sample_image.tif
  ```

## Dependencies

- PyTorch
- OpenCV
- NumPy
- NetworkX
- Rasterio
