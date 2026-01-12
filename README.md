# VAE-Based Music Clustering: Three-Level Task Implementation

## Overview

This project implements unsupervised learning for music clustering using Variational Autoencoders (VAE) with MFCC audio features. Three progressive tasks build from basic to advanced implementations:

- **Easy Task**: Basic VAE with K-Means clustering and PCA baseline comparison
- **Medium Task**: Enhanced VAE with hybrid audio+lyrics features and multiple clustering algorithms
- **Hard Task**: Beta-VAE/CVAE with multi-modal fusion, advanced metrics, and baseline comparisons

---

## Directory Structure

```
CSE425_Music_Clustering/
├── data/
│   ├── audio/                # Music files (organized by genre)
|   |── lyrics/               # Lyrics 
│   ├── features.npz          # Extracted audio features (optional)
│   └── processed/            # Processed features CSV
│
├── notebooks/
│   └── exploratory.ipynb     # Data exploration and analysis
│
├── src/
│   ├── easy_task.py          # Basic VAE + K-Means clustering
│   ├── medium_task.py        # Enhanced VAE + hybrid features
│   ├── hard_task.py          # Beta-VAE/CVAE + advanced evaluation
│   ├── dataset.py            # GTZAN feature extraction utility
│
├── results/
│   ├── models/               # Trained model weights
│   ├── easy_task/           # Easy task outputs
│   ├── medium_task/         # Medium task outputs
│   └── hard_task/           # Hard task outputs
│
├── requirements.txt          # Python dependencies
└── README.md               

## Datasets

1. **GTZAN Dataset:** Download from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 
2. **Genius Lyrics:** Download from [Kaggle](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
