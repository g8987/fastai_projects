Movie Recommendation System — Implementing Probabilistic Matrix Factorization (PMF) and Neural Collaborative Filtering (NCF) from Scratch
📖 Overview
This project demonstrates the end-to-end implementation of a Movie Recommendation System using Probabilistic Matrix Factorization (PMF) and its neural extension, Neural Collaborative Filtering (CollabNN), built completely from scratch using FastAI and PyTorch.
The goal of this project is to move beyond pre-built libraries and understand how user–item interactions can be modeled mathematically and then extended into a learnable neural network architecture.

🚀 Key Concepts Implemented
Collaborative Filtering: Learning user–item relationships using rating data
Probabilistic Matrix Factorization (PMF): Modeling user–movie ratings as dot products of latent feature vectors under Gaussian noise assumptions
Embeddings: Representing users and movies as learnable continuous vectors
Bias Terms: Capturing user tendencies and movie popularity
PCA Visualization: Exploring learned embedding spaces and movie clusters
Cosine Similarity: Finding movies most similar to a given title
Neural Collaborative Filtering (CollabNN): Replacing simple dot products with nonlinear layers to model complex interactions
