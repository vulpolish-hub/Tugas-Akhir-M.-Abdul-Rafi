# Cloud Removal Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the preprocessing and evaluation notebooks with structural visualizations and stratified statistics, and create a comprehensive root README for repository documentation.

**Architecture:** We will load local multi-temporal sample tensors, extract and visualize individual channels (Sentinel-2, Sentinel-1, probability maps, masks, target composites), calculate stratified MAE metrics across three cloud density levels (Light, Medium, Heavy), and draft a detailed README file explaining the end-to-end cloud removal pipeline.

**Tech Stack:** Python, Jupyter Notebook (JSON), PyTorch, NumPy, Matplotlib, Seaborn, Pandas, Scipy.

---

### Task 1: Preprocessing Visualizations

**Files:**
- Modify: `dataset_preprocessing_info.ipynb`

- [ ] **Step 1: Write Python script cell to load a single sample and plot the five visual channels**
  We will add a cell plotting: Sentinel-2 RGB, Sentinel-1 False Color, Spectral Cloud Probability Map, Soft Cloud Mask, and Pseudo Ground Truth.
- [ ] **Step 2: Verify the notebook is a valid JSON and compiles without errors**
  Run verification to check formatting and render properties.
- [ ] **Step 3: Commit**
  Run: `git add dataset_preprocessing_info.ipynb` and commit.

---

### Task 2: Evaluation stratified analysis & Visual Comparison Grid

**Files:**
- Modify: `model_evaluation_inference.ipynb`

- [ ] **Step 1: Write Python code cell for Cloud Stratified Analysis (Light, Medium, Heavy)**
  We will compute average MAE for Baseline and GAN models sliced by mask coverage thresholds (<0.15, 0.15-0.35, >=0.35).
- [ ] **Step 2: Write Python code cell for side-by-side reconstruction comparison grid**
  Plot the comparison layout for 2 test samples: Input, GT, Baseline, GAN, Cloud Mask.
- [ ] **Step 3: Verify the notebook is a valid JSON and compiles without errors**
  Run verification to check formatting and render properties.
- [ ] **Step 4: Commit**
  Run: `git add model_evaluation_inference.ipynb` and commit.

---

### Task 3: Comprehensive Project Documentation (README)

**Files:**
- Create: `README.md` [NEW]

- [ ] **Step 1: Create README.md at the repository root**
  Draft a detailed guide in Indonesian explaining preprocessing, Baseline training, GAN training, evaluation metrics, t-test, scatter plot, and visual trade-offs.
- [ ] **Step 2: Commit**
  Run: `git add README.md` and commit.
