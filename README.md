# Homomorphic Music Retrieval

**Efficient encrypted similarity search for music embeddings using 
AHE/FHE**

## Overview
This repository contains the implementation for encrypted music retrieval 
using:
- Microsoft TenSEAL (CKKS scheme) for FHE/AHE encryption
- YAMNet for generating music embeddings (128-1024 dimensions)
- Milvus for vector storage and retrieval
- MagnaTagATune dataset (1,000 sampled MP3s)

## Key Features
- Supports both encrypted database and encrypted query settings
- Optimized AHE implementation for faster dot products
- Memory-efficient operations for ciphertext-plaintext interactions

## Usage
The code compares FHE vs. AHE performance across different vector lengths 
(128, 256, 512, 1024) in an AWS EC2 t3.xlarge environment.# 
homomorphic
