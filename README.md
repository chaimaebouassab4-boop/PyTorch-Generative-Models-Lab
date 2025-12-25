# PyTorch-Generative-Models-Lab
The main objective is to master the PyTorch library by building and training deep neural network architectures for Generative AI. 

# Lab 4: Deep Learning - Generative Models (AE, VAE & GANs)

https://colab.research.google.com/drive/1XFXLzXbTPgEBsnnK70UaBQxyvR0f_yNT?usp=sharing

# PyTorch Generative Models Lab

Overview
--------
This repository contains experiments implementing and comparing generative models using PyTorch: an Autoencoder (AE), a Variational Autoencoder (VAE) and a basic GAN. The primary code is in `atelier4.py` and the repository includes output images showing training progress, latent space visualizations, reconstruction quality, and generated samples.

Key files
---------
- `atelier4.py` — main experiment script (model definitions, training loops and visualization routines).
- Images (examples and evaluation results):
  - [mnist image.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/mnist%20image.png) — example input images (MNIST).
  - [auto encoder build.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/auto%20encoder%20build.png) — architecture/build of the autoencoder.
  - [latent space.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/latent%20space.png) — latent space visualization (AE/VAE).
  - [VISUALIZING LATENT SPACES.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/VISUALIZING%20LATENT%20SPACES.png) — more latent space visualizations or traversals.
  - [training vae.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/training%20vae.png) — VAE training curve (loss).
  - [quality of reconstruction.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/quality%20of%20reconstruction.png) — AE/VAE reconstruction examples.
  - [EVALUATION COMPARING AE vs VAE.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/EVALUATION%20COMPARING%20AE%20vs%20VAE.png) — comparison of AE vs VAE (metrics or qualitative).
  - GAN outputs: [GAN 1.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/GAN%201.png), [GAN 2.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/GAN%202.png), [GAN 3.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/GAN%203.png), [GAN4.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/GAN4.png), [GAN 5.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/GAN%205.png).
  - [completed tasks.png](https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/completed%20tasks.png) — checklist / project status.



## 📌 Overview
This laboratory work is part of the **Master in MBD (Data Science)** program at the Faculty of Sciences and Techniques of Tangier (FSTT). The main objective is to master the **PyTorch** library by building and training deep neural network architectures for Generative AI.

## 🎯 Objectives
* Build and train **Auto-encoders (AE)** and **Variational Auto-encoders (VAE)**.
* Analyze and plot the **Latent Space** of generative models.
* Implement **Generative Adversarial Networks (GANs)** to generate synthetic data.

## 📂 Laboratory Structure

### Part 1: AE and VAE
**Dataset:** [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
1.  **Auto-encoder (AE):** Implementation of Encoder/Decoder architecture and hyper-parameter optimization.
2.  **Variational Auto-encoder (VAE):** Implementation using the reparameterization trick and KL Divergence.
3.  **Evaluation:** Plotting Loss and KL divergence to conclude on model performance.
4.  **Latent Space Visualization:** Mapping the latent space to understand how the model clusters different digits.

### Part 2: GANs (Generative Adversarial Networks)
**Dataset:** [Abstract Art Gallery](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery)
1.  **Architecture:** Definition of the Generator and Discriminator networks.
2.  **Setup:** Configuration of Loss functions (BCE), Optimizers (Adam), and Data Loaders.
3.  **Training:** GPU-accelerated training process.
4.  **Evaluation:** Analyzing the competition between the Generator and Discriminator through loss plots.
5.  **Data Generation:** Producing new "Abstract Art" and comparing quality with the original dataset.

---

## 🛠️ Tools & Technologies
* **Framework:** PyTorch
* **Environment:** Google Colab / Kaggle (GPU enabled)
* **Version Control:** Git / GitHub

## 📈 Key Findings & Synthesis
what I  learned regarding latent space continuity in VAEs versus AEs, and the stability challenges of training GANs :

**Instructor:** Pr. ELAACHAK LOTFI  
**Institution:** Université Abdelmalek Essaadi, FST Tanger  
**Department:** Computer Engineering - Master MBD
