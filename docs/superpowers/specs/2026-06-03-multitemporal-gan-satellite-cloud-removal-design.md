# Design Spec: Multi-Temporal GAN Satellite Cloud Removal

This document outlines the architecture, data flow, and training strategy for comparing a Multi-Temporal Generative Adversarial Network (GAN) against the baseline Multi-Temporal ResUNet regression model.

## 1. Objectives
* Incorporate a GAN framework matching the thesis proposal requirements.
* Design a stable ResNet-based Discriminator to prevent mode collapse.
* Create a direct, fair comparison with the baseline Multi-Temporal ResUNet (Version 14).
* Establish a comparative study section in BAB IV of the thesis.

## 2. Architecture Design

### 2.1 Generator
* **Model**: `MultiTemporalResUNet`
* **Input**: 64 channels
  * 4 frames of Sentinel-2 (13 channels each) -> 52 channels
  * 4 frames of Cloud Probabilities (1 channel each) -> 4 channels
  * 4 frames of Sentinel-1 SAR (2 channels each) -> 8 channels
* **Output**: 13 channels (restored S2 image at $t_0$)
* **Activation**: Sigmoid at output.

### 2.2 Discriminator
* **Model**: `ResNetDiscriminator`
* **Architecture**: Based on ResNet-18 (using residual downsampling blocks).
* **Input**: 13 channels (real S2 ground truth or generated output).
* **Spectral Normalization**: Applied to all convolutional layers.
* **Output**: 1 channel (scalar probability of real vs fake) using a Sigmoid activation at the final layer.

```mermaid
graph TD
    S2[Sentinel-2 (4x13)] --> GenInput
    CP[Cloud Prob (4x1)] --> GenInput
    S1[Sentinel-1 SAR (4x2)] --> GenInput
    GenInput[Generator Input (64 channels)] --> Generator[Generator: MultiTemporalResUNet]
    Generator --> Pred[Restored Image: Generated (13 channels)]
    Pred --> Disc[Discriminator: ResNet-18 + Spectral Norm]
    GT[Ground Truth: Real (13 channels)] --> Disc
    Disc --> Out[BCE Loss: Real vs Fake]
```

## 3. Loss Functions

### 3.1 Discriminator Loss
Standard Binary Cross Entropy (BCE) Loss with Logits:
$$\mathcal{L}_D = -\mathbb{E}[\log(D(y))] - \mathbb{E}[\log(1 - D(G(x)))]$$
Where:
* $y$ is the Ground Truth (real).
* $G(x)$ is the generated image (fake).

### 3.2 Generator Loss
A composite loss including L1, Gradient, and Adversarial GAN loss:
$$\mathcal{L}_G = \alpha \mathcal{L}_{L1} + \beta \mathcal{L}_{grad} + \gamma \mathcal{L}_{adv} + \delta \mathcal{L}_{copy}$$
Where:
* $\mathcal{L}_{L1}$ contains the whole, masked, raw, and heavy-cloud L1 losses.
* $\mathcal{L}_{grad}$ is the Spatial Gradient MAE.
* $\mathcal{L}_{copy}$ is the penalty for modifying pixels outside the cloud mask.
* $\mathcal{L}_{adv} = -\mathbb{E}[\log(D(G(x)))]$ is the adversarial loss pushing the generator to produce sharp, realistic textures.

## 4. Verification Plan

### 4.1 Automated Metrics
* **PSNR**: Peak Signal-to-Noise Ratio (aiming to match or be close to baseline ~37 dB).
* **MAE**: Mean Absolute Error on cloudy and heavy-cloud regions.
* **Training stability**: Monitored via generator/discriminator loss curves.

### 4.2 Manual Verification
* Visual inspection of Generated images for high-frequency sharp textures vs baseline smooth images.
* Confirm no mode collapse or severe artifacts in output.
