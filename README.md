# CS3807 – Deep Learning Laboratory (Academic Year:  2026–27)


This repository contains Deep Learning laboratory experiments implemented using **Python** and **Jupyter Notebooks (`.ipynb`)**. All notebooks are designed to run seamlessly on **Google Colab**.

## Experiments

### [**Lab 1** - Implementation of a Single Layer Perceptron for Binary Classification](./Lab1_SLP)
### [**Lab 2** - Implementation of a Multi-Layer Perceptron (MLP) for Multi-Class Image Classification](./Lab2_MLP)
### [**Lab 3** - Implementation of Convolutional Neural Networks (CNNs) for Image Classification](./Lab3_CNN)
### [**Lab 4** - Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning](./Lab4_TransferLearning)
### [**Lab 5** - Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation](./Lab5_MobileNetV2)
## Prerequisites

- Python 3.9+
- Google Colab (or) Jupyter Notebook

Required Python libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn

## Setup

### Option 1: Google Colab

1. Open **Google Colab**.
2. Upload the desired `.ipynb` notebook from this repository.
3. Run the notebook cells sequentially.

Alternatively, clone this repository directly in Colab:

```bash
!git clone https://github.com/ssvibitha/DeepLearningLab_2026-27.git
```

Navigate to the notebook and run all cells sequentially.

### Option 2: Local Environment

If you prefer to run the notebooks locally:

```bash
git clone https://github.com/ssvibitha/DeepLearningLab_2026-27.git
cd DeepLearningLab_2026-27

pip install tensorflow keras numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

## Notes

- All notebooks are executed on **Google Colab**.
- GPU acceleration can be enabled from:
  - **Runtime → Change runtime type → GPU**
- Some experiments may require downloading datasets automatically during execution.
