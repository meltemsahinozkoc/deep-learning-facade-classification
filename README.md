# Deep Learning for Building Exterior Cladding Classification Using Pre-trained CNNs

This project applies **transfer learning** using **ResNet50** and **InceptionV3** to classify **building façade cladding materials** from labeled Google SVIs. 

## Objective

Automatically classify exterior cladding types—such as **Brick**, **Concrete**, **Curtain-Wall**, **Mixed**, **Others**, and **Stone**—to support scalable building stock analysis for energy and urban modeling.

## Dataset

- Source: [Wang et al., 2024 – Building Façade Dataset](https://doi.org/10.1016/j.dib.2024.110885)
- Cities: **London** (for training) and **Scotland** (for generalization testing)
- Format: Image folders with class-wise subdirectories

## Models

- **ResNet50** and **InceptionV3** with frozen base layers and custom dense heads
- Trained separately on **unaugmented** and **augmented** datasets

## Key Results

| Model       | Dataset     | Test Acc. | 
|-------------|-------------|-----------|
| ResNet50    | Augmented   |   68.2%   | 
| InceptionV3 | Augmented   | **70.4%** |

## Notebooks

- `final_project_unaugmented.ipynb`
- `final_project_augmented.ipynb`

## Next Steps

- Fine-tune base layers
- Try **Vision Transformers (ViT)** or **Swin Transformers** [Liu et al., 2025](https://doi.org/10.1016/j.enbuild.2025.115457)
- Work on domain adaptation.

## Requirements

Python 3.9 · TensorFlow · NumPy · Matplotlib · scikit-learn · seaborn  
(Optimized for Apple Silicon with `tensorflow-metal`) - 1 GPU


© 2025 Meltem Sahin Ozkoc – Carnegie Mellon University
