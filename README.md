# Stacked Ensemble Models for Cancer Classification

## Overview
This project implements a stacked ensemble learning framework for cancer classification using high-dimensional gene expression data. The goal is to improve predictive performance while maintaining biological interpretability by integrating machine learning models with gene-level and pathway-level analysis.

The framework combines multiple base learners into a meta-model and identifies biologically meaningful genes and pathways that contribute to cancer subtype classification.

---

## Project Objectives
- Build a stacked ensemble model for cancer classification  
- Improve classification performance over single-model approaches  
- Perform feature selection to identify biologically relevant genes  
- Map selected genes to known biological pathways  
- Provide a reproducible and modular analysis pipeline 

---

## Project Structure
1. data/ 
- Contains four cancer dataset downloaded from CUMIDA database: https://sbcb.inf.ufrgs.br/cumida
- Each dataset consists of gene expression values measured across multiple samples, where rows correspond to genes (probe IDs) and columns represent samples.
2. notebooks/
- Contains Jupyter notebooks used for exploratory data analysis, visualization, and biological interpretation of results.
3. src/
- Contains the main implementation of the project, including:
(1) Training base machine learning models (e.g., SVM)
(2) Building stacked ensemble models
(3) Performing feature selection
(4) Mapping selected genes to biological pathways
4. results/
- includes the project report, summarize results from the analysis.

---

## Methodology

### 1. Data Preparation
Gene expression datasets are cleaned, normalized, and formatted prior to modeling.

### 2. Model Architecture
- **Base learner:**
  - Support Vector Machine (SVM)

- **Meta-learner:**
  - Decision Tree  
  - Random Forest  
  - Logistic Regression  

### 3. Feature Selection
Recursive feature selection is used to identify genes that contribute most strongly to classification performance.

### 4. Biological Interpretation
Selected top 10 and 20 genes are mapped to known biological pathways to provide interpretable biological insight.

Author: Tianyi Gao (Collaborator: Audrey Qian)
titigao@bu.edu
Boston University