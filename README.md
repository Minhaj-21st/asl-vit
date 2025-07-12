# 🧠 ASL-ViT: American Sign Language Classification with Vision Transformer

## Overview

**ASL-ViT** is a deep learning project for classifying American Sign Language (ASL) hand gestures using the Vision Transformer (ViT) architecture. The model is fine-tuned from the Hugging Face `google/vit-base-patch16-224-in21k` checkpoint to recognize ASL alphabet signs from images. This project aims to support assistive technology, education, and accessibility.

---

## ✨ Features
- Vision Transformer (ViT) backbone for image classification
- Fine-tuning on ASL alphabet dataset
- Data augmentation and class imbalance handling
- Training and evaluation scripts
- SLURM support for HPC clusters

---

## 📁 Project Structure

```
├── ViT_ASL_Model.py      # Main training and evaluation script
├── A.slurm              # SLURM job submission script (for HPC)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── data/                # [Place your dataset here]
```

---

## 🛠️ Requirements
- Python 3.8 or newer
- See `requirements.txt` for all dependencies

**Key packages:**
- torch
- torchvision
- transformers
- scikit-learn
- timm

---

## ⚙️ Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd asl-vit
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Dataset
- The dataset is **not included** due to size constraints.
- Download the [ASL Alphabet Dataset by Debashish Sau](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle.
- Extract the dataset and place the folder inside your project directory as `data/` (or update the path in `ViT_ASL_Model.py`).

---

## 🚀 Usage

### Train and Evaluate the Model
Run the training script:
```bash
python ViT_ASL_Model.py
```
This will:
- Load and preprocess the dataset
- Fine-tune the Vision Transformer
- Evaluate and print performance metrics (accuracy, precision, recall, F1, classification report)

### ⚡ SLURM Support (HPC)
To train on an HPC cluster:
```bash
sbatch A.slurm
```
Edit `A.slurm` as needed for your cluster environment.

---

## 📊 Results
After training, the script prints evaluation metrics including accuracy, precision, recall, F1 score, and a full classification report.

---

## 📄 License
This project is licensed under the MIT License.

---
<<<<<<< HEAD

## 🤝 Acknowledgements
- Hugging Face Transformers
- [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

© 2025 Hafijur Raman
=======
](https://github.com/Minhaj-21st/asl-vit.git)
>>>>>>> 8f75299ab4fb5eb7e313a0657282740cc84eeda4
