# StyleGAN — Multi-Category Image Generation

A StyleGAN generator trained from scratch on a **balanced, multi-category image set** (dogs, cars, and food) to produce diverse 128×128 images. The goal was to test whether a single StyleGAN model can learn a genuinely multimodal distribution — covering three visually distinct categories at once — rather than collapsing toward any one of them.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![GAN](https://img.shields.io/badge/Model-StyleGAN-8A2BE2?style=flat-square)

---

## Overview

The training set is a balanced blend of three categories — **dogs**, **cars**, and **food** — combined into a single dataset of 128×128 color images. Because the categories are mixed and unlabeled during training, the generator has to model a wide variety of content from one latent space. Sampling from the trained model produces images spanning all three categories, showing that the network learns the full distribution instead of favoring a single mode.

- **Model** — StyleGAN generator + discriminator, implemented in PyTorch
- **Data** — balanced multi-category set (dogs / cars / food), normalized 128×128 RGB
- **Output** — diverse generated samples across all three categories from a single model
- **Artifacts** — trained generator and discriminator weights are included for inference

## Repository

```
StyleGAN/
├── StyleGAN.ipynb              # model, training loop, and sampling
├── stylegan_generator.pth      # trained generator weights
├── stylegan_discriminator.pth  # trained discriminator weights
└── StyleGAN Part (1).pdf       # report: approach, dataset, and results
```

## Running

```bash
pip install torch torchvision numpy matplotlib
```

- Open `StyleGAN.ipynb` to train from scratch, or load `stylegan_generator.pth` to generate samples directly.
- Prepare the dataset as 128×128 images across the dog / car / food categories and point the notebook's data loader at it.

## Notes

Trained weights are committed so generation can be reproduced without retraining. See `StyleGAN Part (1).pdf` for the full write-up of the dataset construction, architecture, and generated results.
