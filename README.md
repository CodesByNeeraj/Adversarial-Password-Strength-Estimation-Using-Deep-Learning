# Adversarial-Password-Strength-Estimation-Using-Deep-Learning
Can Generative AI model the probability distribution of human-created passwords better than traditional rule-based algorithms?

## Abstract
Passwords are still the most common way people protect their online accounts, but many
users continue to choose predictable passwords. Security tools like Hashcat try to guess these
passwords using some password generation rules such as adding numbers, changing letters to
symbols, or capitalizing words. These rules work well for simple patterns, but they cannot
adapt to new trends or understand the complex structure behind how humans create
passwords. Recent research showed that a generative adversarial network (PassGAN) can
learn password patterns directly from leaked datasets like RockYou, improving password
guessing without relying on traditional rules. However, GANs have known issues such as
unstable training and mode collapse, which can limit their ability to detect complex password
patterns. To address this gap, we aim to build a Diffusion-based password generator
(PassDiffusion) and use PassGAN as the main baseline. Since, diffusion models have
recently become very successful at generating high-quality and diverse samples in many
fields and may overcome the problems seen in GANs. The goal is to train PassDiffusion on
the RockYou dataset and compare it against PassGAN and a simple rule-based method. By
testing these models on a hidden test set, we hope to show whether diffusion models can
better capture real human password behavior and help improve password strength auditing.

## Goals
### 1. Compare Against Baselines
Benchmark against PassGAN (GAN baseline) and a simple rule-based method using match rate on a held-out test set.

### 2. Build PassDiffusion
Train a diffusion-based password generator on the RockYou dataset combined with latest dataset leaks to learn human password distributions.

### 3. Final product
- Comparative analysis of diffusion models (PassDiffusion) over GAN model (PassGAN)
- Password Strength Estimation WebApp and Research Paper (if findings are substantial enough that warrants a research paper)



