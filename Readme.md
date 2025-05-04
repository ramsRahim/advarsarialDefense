# Test-Time Adversarial Defense for Vision-Language Models

This repository contains the code for our project **"Test-Time Adversarial Defense in Vision-Language Models via Dual Adaptive Caching"**. The goal of this project is to build a lightweight, data-free test-time defense against adversarial attacks for vision-language models (VLMs) like CLIP. 

Our implementation is motivated by the [TDA framework](https://github.com/kdiAAA/TDA.git), that does zero-shot classification accuarcy increase of different domains. We propose a new solution for Test-time adversarial defense by incorporating an MLP-based adversarial detector and a dual-cache system for logit correction of pre-trained CLIP.

---

## Setup Instructions

First, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```


## Train Adversarial Sample Detector

To train the MLP for detecting adversarial and clean samples at test time, run the following script

```bash
python mlp_adversarial_detector.py
```

This will train and save the detector locally

## Run Test-time defense

Our current implementation is focused on defending FGSM attack on CLIP zero-shot model. To run the test time defense code, please run the following command

```bash
bash scripts/run_fgsm_rn50.sh
```

# Team Members
Rahim Hossain (ramsRahim),
Mohaiminul Al Nahian (alnahian37)

# Contribution
- Both contributed equally to reproduce main result from TDA paper
- For literature survey, Rahim contributed on Test-time adaptation and Nahian contributed on adversarial defense
- Both contributed equally on theory and hypothesis development fot the project
- Both contributed for setting up code base for the test-time adversarial defense
- Both prepared the presentation and write the report
- Overall contribution: Rahim 50%, Nahian 50%
