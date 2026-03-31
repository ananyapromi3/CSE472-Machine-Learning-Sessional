# CSE472 Machine Learning Sessional

Coursework, practice code, and reference materials for the `CSE 472 Machine Learning Sessional` repository. The repository combines offline assignments, preprocessing notebooks, classical machine learning implementations, and CNN-based deep learning experiments.

## Repository Overview

```text
.
├── Offline1/
│   ├── 2005079/
│   │   ├── 2005079.ipynb
│   │   └── 2005079_Report.pdf
│   ├── preprocessing_intro_materials/
│   │   ├── iris_classification.ipynb
│   │   ├── Feature_non_numeric.ipynb
│   │   ├── iris.csv
│   │   └── melb_data.csv
│   └── July25_CSE472_Assignment1.pdf
├── Offline 2/
│   ├── 2005079/
│   │   ├── 2005079.ipynb
│   │   └── ML_Offline_2.pdf
│   └── CSE_472_Assignment_2_DT.pdf
└── Online/
    ├── cnn.py
    ├── pytorch-cheatsheet-en.pdf
    └── cnn-online-references/
```

## Contents

### Offline 1
Focuses on data understanding, preprocessing, and neural-network-based classification.

Main work:
- Exploratory data analysis with `pandas`, `matplotlib`, and `seaborn`
- Data cleaning, including missing-value handling and duplicate removal
- Encoding categorical variables using techniques such as label encoding and one-hot encoding
- Feature scaling with `StandardScaler`
- Correlation analysis and feature selection
- Feed-forward neural network experiments in PyTorch
- Evaluation using metrics such as accuracy, precision, F1-score, ROC-AUC, and confusion matrix

Important files:
- `Offline1/2005079/2005079.ipynb`: main submission notebook
- `Offline1/2005079/2005079_Report.pdf`: report for the assignment
- `Offline1/preprocessing_intro_materials/`: introductory notebooks and sample datasets

### Offline 2
Focuses on tree-based machine learning methods and ensemble learning.

Main work:
- Decision Tree implementation and evaluation
- Random Forest experiments
- Extremely Randomized Trees (Extra Trees) experiments
- Metric-based comparison of model behavior
- Visualization support using `matplotlib` and `seaborn`

Important files:
- `Offline 2/2005079/2005079.ipynb`: main notebook for the assignment
- `Offline 2/2005079/ML_Offline_2.pdf`: related assignment material
- `Offline 2/CSE_472_Assignment_2_DT.pdf`: problem statement/reference PDF

### Online
Focuses on CNN practice and architecture design exercises in PyTorch.

Main work:
- `Online/cnn.py`: a basic convolutional neural network trained on MNIST using PyTorch
- `Online/cnn-online-references/20/A1/`: CIFAR-10 experiments, including Network in Network style models
- `Online/cnn-online-references/21/A1-A2/Question/`: custom ResNet-style exercise and solution
- `Online/cnn-online-references/21/B1-B2/Question/`: custom optimizer and Mini-Inception exercise and solution
- `Online/cnn-online-references/21/C1-C2/Question/`: CNN architecture design exercise and solution
- `Online/pytorch-cheatsheet-en.pdf`: quick PyTorch reference material

## Tech Stack

This repository primarily uses:
- Python 3
- Jupyter Notebook
- PyTorch and TorchVision
- NumPy and pandas
- scikit-learn
- Matplotlib and seaborn

## Suggested Environment Setup

Create a virtual environment and install the common dependencies used across the notebooks and scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter notebook numpy pandas matplotlib seaborn scikit-learn torch torchvision tqdm
```

Depending on your platform and hardware, you may want to install PyTorch from the official selector instead of the generic `pip install torch torchvision` command.

## How to Use

### Open the notebooks

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Then open any of the notebooks under `Offline1/` or `Offline 2/`.

### Run the CNN example

From the repository root:

```bash
python3 Online/cnn.py
```

This script downloads the MNIST dataset automatically and saves a trained model checkpoint as `model.ckpt`.

### Run the CIFAR-10 reference models

Examples:

```bash
python3 "Online/cnn-online-references/20/A1/2005079.py"
python3 "Online/cnn-online-references/20/A1/2005001.py"
```

These scripts download CIFAR-10 automatically when required.

## Notes

- Many files in `Online/cnn-online-references/` are practice problems and reference solutions rather than polished library code.
- Some notebook cells rely on external data sources or notebook execution order.
- The repository is organized as coursework material, so scripts are mostly standalone and do not share a package structure.
- Paths containing spaces, such as `Offline 2/`, should be quoted in terminal commands.

## Who This Repository Is For

This repository is useful for:
- students taking `CSE 472 Machine Learning Sessional`
- anyone revising core preprocessing workflows in Python
- learners practicing PyTorch CNN implementations and custom deep learning architectures
- readers comparing classical ML models with neural-network-based approaches

## License

No license file is currently included in this repository. Unless stated otherwise by the author or course policy, treat the contents as academic coursework material.
