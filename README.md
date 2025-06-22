# 🧠 ASL-ViT: American Sign Language Classification with Vision Transformer

![ASL Sample A](https://upload.wikimedia.org/wikipedia/commons/4/49/ASL_A.svg)
![ASL Sample B](https://upload.wikimedia.org/wikipedia/commons/3/3b/ASL_B.svg)

---

## 📌 Project Description

**ASL-ViT** is a deep learning project focused on classifying **American Sign Language (ASL)** hand gestures using the **Vision Transformer (ViT)** architecture. The model is fine-tuned from the pretrained Hugging Face transformer `google/vit-base-patch16-224-in21k` to effectively recognize ASL alphabet signs from images.

This project aims to build a lightweight, accurate, and scalable ASL recognition system for assistive technology, educational tools, and accessibility enhancement.

---

## 📂 Project Structure

.
├── ViT_ASL_Model.py # Main training and evaluation script
├── A.slurm # SLURM job submission script (for HPC usage)
├── requirements.txt # Python package dependencies
├── README.md # Project documentation
└── data/ # [You will place the dataset here]

yaml
Copy
Edit

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

bash
Copy
Edit
python ViT_ASL_Model.py
This will:

Load and preprocess the dataset

Fine-tune the Vision Transformer

Evaluate and print performance metrics



⚡ SLURM Support
To train on an HPC cluster, you can use the provided SLURM script:

bash
Copy
Edit
sbatch A.slurm
Make sure to edit the script to match your compute environment (e.g., modules, job name, time, etc.).

📈 Future Work
Expand to dynamic gesture classification (ASL words or sentences)

Integrate real-time webcam inference

Deploy as a web or mobile app using ONNX/TensorRT

📄 License
This project is licensed under the MIT License.

© 2025 Dipok Deb

🤝 Acknowledgements
Hugging Face Transformers

ASL Alphabet Dataset on Kaggle

yaml
Copy
Edit

---

You're all set! Paste this into your `README.md` file and push to GitHub. Let me know if you want a markdown preview or live demo added next.








