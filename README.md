# Adversarial-Password-Strength-Estimation-Using-Deep-Learning
Can Generative AI model the probability distribution of human-created passwords better than traditional rule-based algorithms?


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
