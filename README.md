# PyTorch-Generative-Models-Lab
The main objective is to master the PyTorch library by building and training deep neural network architectures for Generative AI. 

# Lab 4: Deep Learning - Generative Models (AE, VAE & GANs)

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
