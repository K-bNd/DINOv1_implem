# README.md for DINOv1 Repository (redo from [group project](https://github.com/Coartix/DNN_Dino/))

## Overview
This repository implements a DINO (Distillation of Self-Supervised Learning) model, focusing on self-supervised learning techniques for computer vision. It's based on Vision Transformers (ViTs) and includes custom neural network architectures and data augmentation strategies.

## Main Components
- `Trainer.py`: Manages model training, including setup and execution.
- `train.py`: Main script to configure and start the training process.
- `eval.py`: Main script to evaluate the model on kNN accuracy.
- `utils.py`: Provides utility functions and classes for model training.
- `models/DINO.py`: Contains the DINO model implementation.
- `models/DINO_head.py`: Contains the DINO head implementation.
- `models/DINO_loss.py`: Contains the DINO loss implementation. (TODO)
- `configs/`: YAML configuration files for setting up models and datasets and Pydantic models for model validation.

## Training and Usage
To train the DINO model:
1. Set up your environment and install required dependencies using [Poetry](https://python-poetry.org/docs/basic-usage/#installing-dependencies)
2. Configure your model and dataset paths in the YAML files in the `configs` folder.
3. Run `train.py` with the desired configuration file.

## Evaluation
To evaluate the DINO model:
1. Set up your environment and install required dependencies using [Poetry](https://python-poetry.org/docs/basic-usage/#installing-dependencies)
2. Pick the pretrained model you want to evaluate in `training_output`
3. Run the `eval.py` with the desired configuration from the CLI command

## References
For more information on the underlying concepts and methodologies, refer to the original paper: [Self-supervised Learning of Pretext-invariant Representations](https://arxiv.org/pdf/2104.14294.pdf).
