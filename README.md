# 🧠 Medical Image Enhancement using GANs

**🚧 Project Status: In Progress**

This repository contains the implementation of a Medical Image Enhancement system using Generative Adversarial Networks (GANs). The aim is to enhance low-quality or noisy medical images (e.g., retinal scans and brain MRIs) to aid in clinical diagnostics and support healthcare professionals. The project is implemented using PyTorch and deployed via Streamlit for easy interaction.

---

## 🗂️ Directory Structure

```
Project/
│
├── data/
│   ├── processed/
│   │   ├── brain_mri/
│   │   ├── stare/
│   │   ├── stare_aug/
│   │   └── stare.zip
│   ├── raw/
│   ├── test/
│   └── dataset_info.txt
│
├── logs/
├── notebooks/
├── reports/
├── results/
│   ├── enhanced_sample.jpg
│   └── saved_models/
│
├── src/
│   ├── deployment/
│   ├── evaluation/
│   ├── models/
│   │   ├── agumentation.py
│   │   ├── app.py
│   │   ├── cuda test.py
│   │   ├── discriminator.py
│   │   ├── generator.py
│   │   ├── loss_functions.py
│   │   └── trainer.py
│   ├── preprocessing/
│   ├── tests/
│   │   └── test_model.py
│   └── utils/
└── utils/
```

---

## 💡 Project Workflow

- **Data Loading and Augmentation:** Using STARE and brain MRI datasets (3000+ images, `.nii` format from Mendeley Data).
- **Preprocessing:** Standard normalization and resizing.
- **Model Design:** Custom-built Generator and Discriminator using PyTorch.
- **Training:** Progressive GAN training with custom loss functions.
- **Evaluation:** Quality visualization and test scripts.
- **Deployment:** Real-time enhancement interface using Streamlit.
- **Optimization:** Tailored for NVIDIA GeForce MX330 (low VRAM), memory-efficient loading and inference.

---

## 🚀 How to Run

### 🧱 Setup

```bash
git clone https://github.com/your-username/medical-image-enhancement-gan.git
cd medical-image-enhancement-gan
pip install -r requirements.txt
```

### ▶️ Launch Streamlit App

```bash
cd src/models
streamlit run app.py
```

---

## 🧪 Test the Model

```bash
cd src/tests
python test_model.py
```

---

## 📸 Sample Output

Here’s an example of a successfully enhanced medical image:

![Enhanced Medical Image](results/enhanced_sample.jpg)

> *Original image is low-resolution and noisy, while the enhanced output is clearer and more defined using GAN.*

---

## 🛠️ Technologies Used

- Python 3.x
- PyTorch
- Streamlit
- OpenCV, NumPy, Matplotlib
- Librosa (for audio, if extended)
- TorchVision
- PIL
- Pickle

---

## 📚 Resources & Dataset

- STARE Dataset (Augmented)
- Brain MRI Dataset (Mendeley Data - Checksum: `e2f95dc91251068a73c78873291bd2eb57d116c877b4f2d14eab34c5f45ad0e0`)
- Streamlit for deployment

---

## 🧠 Learned Concepts

This project has helped me strengthen my understanding of:

- Deep learning with GANs (Generator/Discriminator training dynamics)
- Custom loss function design
- Data preprocessing for medical image formats (DICOM, .nii)
- Streamlit deployment and interactive UI
- Efficient GPU memory usage with limited resources (NVIDIA GeForce MX330)
- Packaging and modular code structure for scalable ML projects

---

## 📌 To-Do (Next Steps)

- [ ] Add evaluation metrics (PSNR, SSIM)
- [ ] Add confusion matrix and detailed logging
- [ ] Improve UI with output comparison
- [ ] Upload demo video/gif to README
- [ ] Publish Streamlit cloud deployment (if resources allow)

---

## 🔗 Author

**Shardul More**  
Final Year, Sanjay Ghodawat University  
Actively seeking a 6-month Data Science internship (Jan–June 2025)  
📧 Reach me via LinkedIn | GitHub | Email

---

> **Note:** This project is part of my AI/ML research exploration and will continue to evolve with further enhancements and clinical evaluations.

