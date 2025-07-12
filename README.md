[# 🧠 ASL-ViT: American Sign Language Classification with Vision Transformer

---

## 📌 Project Description

**ASL-ViT** is a deep learning project focused on classifying **American Sign Language (ASL)** hand gestures using the **Vision Transformer (ViT)** architecture. The model is fine-tuned from the pretrained Hugging Face transformer `google/vit-base-patch16-224-in21k` to effectively recognize ASL alphabet signs from images.

This project aims to build a lightweight, accurate, and scalable ASL recognition system for assistive technology, educational tools, and accessibility enhancement.

---

## 📂 Project Structure

├── ViT_ASL_Model.py # Main training and evaluation script
├── A.slurm # SLURM job submission script (for HPC usage)
├── requirements.txt # Python package dependencies
├── README.md # Project documentation
└── data/ # [You will place the dataset here]

---

## 🛠️ Requirements

- Python 3.8 or newer  
- Install all required packages via:

```bash
pip install -r requirements.txt

Core dependencies:

torch

torchvision

transformers

scikit-learn

See full list in requirements.txt

🚀 Usage
Download the dataset
The dataset is not included in this repository due to size constraints.

📥 Download from Kaggle:
👉 ASL Alphabet Dataset by Debashish Sau

After downloading, extract the contents and place the folder inside your project directory as data/.

Train and evaluate the model
Run the training script:
python ViT_ASL_Model.py


This will:

Load and preprocess the dataset

Fine-tune the Vision Transformer

Evaluate and print performance metrics

⚡ SLURM Support
To train on an HPC cluster, you can use the provided SLURM script:
sbatch A.slurm

📄 License
This project is licensed under the MIT License.

© 2025 Hafijur Raman

🤝 Acknowledgements
Hugging Face Transformers
ASL Alphabet Dataset on Kaggle
---
](https://github.com/Minhaj-21st/asl-vit.git)
