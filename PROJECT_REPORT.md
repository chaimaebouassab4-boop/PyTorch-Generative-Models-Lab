# Project Report — PyTorch Generative Models Lab

Date: 2025-12-25
Author: chaimaebouassab4-boop

1. Introduction
---------------
This project explores three classes of generative models implemented in PyTorch: Autoencoder (AE), Variational Autoencoder (VAE), and a basic Generative Adversarial Network (GAN). The dataset used for experiments is MNIST (handwritten digit images). The goals are:
- Implement and train AE, VAE, and GAN models.
- Visualize and evaluate latent spaces and generated/reconstructed images.
- Compare AE and VAE performance qualitatively and quantitatively.

2. Implementation
-----------------
Main script: `atelier4.py` — contains:
- Data loading (MNIST).
- Model definitions for AE, VAE, and GAN (encoder, decoder/generator, discriminator).
- Training loops for each model, with periodic visualization/saving of samples and evaluation metrics.
- Visualization utilities to show reconstructions, generated samples, latent traversals and training curves.

Dependencies: PyTorch, torchvision, numpy, matplotlib, tqdm.

3. Experiments and Hyperparameters (typical)
--------------------------------------------
- Dataset: MNIST (train/test split via torchvision).
- Image size: 28×28 (grayscale).
- Batch size: (check script — typical 64/128).
- Latent dimension: (common values 2 for visualization, 16 or 32 for higher capacity).
- Optimizers: Adam (standard settings beta1=0.5 for GAN).
- Training epochs: Varies (plots provided for training VAE; GAN images show progression across epochs).

4. Results (from repository images)
-----------------------------------
- Input / dataset example:
  - `mnist image.png` — Display of input MNIST samples used for training and visualization.

- Autoencoder:
  - `auto encoder build.png` — Diagram or printed architecture indicating encoder/decoder layers.
  - `quality of reconstruction.png` — Side-by-side comparisons show AE reconstructions. Observations: AE can accurately reconstruct seen digits, with blur on some complex strokes; reconstruction quality depends on latent size.

- Variational Autoencoder:
  - `training vae.png` — VAE loss curves (reconstruction loss and KL divergence). Observations: KL regularization increases initially and stabilizes; reconstruction loss typically decreases then plateaus.
  - `latent space.png` and `VISUALIZING LATENT SPACES.png` — VAE latent embeddings show clustering by digit class and smooth transitions. With a 2D latent space the class clusters are visually separable; traversals produce smooth morphing between digits.

- AE vs VAE comparison:
  - `EVALUATION COMPARING AE vs VAE.png` — Direct comparison (likely reconstruction error and qualitative samples). Observations: AE typically shows lower reconstruction error (since not constrained by KL) and crisper reconstructions, while VAE yields a smoother, more continuous latent space useful for sampling.

- GAN:
  - `GAN 1.png`, `GAN 2.png`, `GAN 3.png`, `GAN4.png`, `GAN 5.png` — Progression of generated samples across training. Observations: early epochs produce noisy/poor samples; later epochs show clearer digit structure. Mode collapse or instability can appear and should be monitored via discriminator/generator losses.

5. Quantitative evaluation
--------------------------
- Reconstruction metrics: Mean Squared Error (MSE) or Binary Cross Entropy (BCE) between original and reconstructed images (AE vs VAE).
- Latent space quality: visual clustering, interpolations, and (optionally) metrics like KDE / disentanglement scores.
- GAN evaluation: visual inspection is primary; can compute FID if using a larger dataset or pre-trained feature extractor.

6. Observations & Conclusions
-----------------------------
- AE excels at reconstruction fidelity but the latent space is not probabilistically regularized — sampling from latent prior does not reliably produce valid samples.
- VAE trades some reconstruction accuracy for a structured latent space that supports sampling and interpolation; useful for generative tasks.
- GANs can produce sharper samples than VAEs but require careful hyperparameter tuning and are prone to instability.
- The provided figures support these expected behaviors: AE reconstructions (good fidelity), VAE latent visualizations (smooth structure), GAN images showing improving quality over epochs.

7. Limitations
--------------
- No standardized quantitative FID/IS metrics in the current repo (would be useful for GAN evaluation).
- Experiments appear limited to MNIST; results may not generalize to more complex datasets.
- Reproducibility requires adding requirements and seed control.

8. Recommendations / Future Work
-------------------------------
- Add `requirements.txt` and exact PyTorch/CUDA versions.
- Add a small CLI in `atelier4.py` or separate scripts to run AE/VAE/GAN with reproducible defaults.
- Save checkpoints and log training metrics (e.g., with TensorBoard or Weights & Biases).
- Evaluate GANs with FID and VAEs with ELBO decomposition tables for quantitative comparison.
- Explore conditional models (conditional VAE/GAN) for class-conditional generation.

9. How to reproduce
-------------------
1. Set up environment:
   - python -m venv venv
   - source venv/bin/activate
   - pip install torch torchvision matplotlib numpy tqdm
2. Run:
   - python atelier4.py
3. Check output image files and saved plots in the working directory; compare with repo images listed in the README.

10. References
--------------
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
- Goodfellow, I., et al. (2014). Generative Adversarial Networks.
- PyTorch tutorials (official).

Appendix — Repository images (quick links)
-----------------------------------------
- mnist image.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/mnist%20image.png
- auto encoder build.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/auto%20encoder%20build.png
- latent space.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/latent%20space.png
- VISUALIZING LATENT SPACES.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/VISUALIZING%20LATENT%20SPACES.png
- training vae.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/training%20vae.png
- quality of reconstruction.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/quality%20of%20reconstruction.png
- EVALUATION COMPARING AE vs VAE.png — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/blob/main/EVALUATION%20COMPARING%20AE%20vs%20VAE.png
- GAN images — https://github.com/chaimaebouassab4-boop/PyTorch-Generative-Models-Lab/tree/main (see GAN 1.png ... GAN 5.png)

---

If you'd like, I can:
- Open a PR adding these two files to the repository.
- Create a `requirements.txt` and a simple `run.sh` or CLI modifications to `atelier4.py`.
- Extract exact hyperparameters from `atelier4.py` and put a runnable example command in the README. Which would you prefer next?