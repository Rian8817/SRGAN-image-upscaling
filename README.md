# 🧠 Super-Resolution GAN (SRGAN) Web Application

This web application generates **high-quality super-resolution images** from low-resolution inputs using **Generative Adversarial Networks (GANs)**.  
Traditional image upscaling methods often produce blurred or distorted results — but with **Deep Learning**, image enhancement becomes optimized and realistic.  
**Super-Resolution GAN (SRGAN)** is one of the most effective architectures for this task.


---

## 📘 Introduction to GANs

**Generative Adversarial Networks (GANs)** are deep learning architectures used for generative modeling — learning data patterns to generate new, realistic examples.  
GANs consist of two main components:

- 🧩 **Generator** – Produces high-resolution images from low-resolution inputs  
- 🕵️ **Discriminator** – Distinguishes between real and generated high-resolution images  

During training, both networks compete:
- The Generator tries to fool the Discriminator,
- The Discriminator improves to better detect fakes.

Through this adversarial process and **backpropagation**, the Generator learns to create realistic, high-resolution outputs.


The SRGAN framework significantly improves image sharpness and detail compared to traditional interpolation methods.  
For more technical insights, refer to the original [SRGAN Research Paper](https://arxiv.org/abs/1809.00219).


---

## ⚙️ Dependencies

| Library | Purpose |
|----------|----------|
| **ISR** | Image Super-Resolution Python library |
| **TensorFlow** | Deep learning framework |
| **Streamlit** | Interactive web interface |
| **numpy** | Numerical computation |
| **Pillow (PIL)** | Image processing |
| **matplotlib** | Visualization |
| **scipy** | Scientific computing |
| **h5py** | HDF5 file format support |
| **tqdm, pyaml, imageio** | Utility modules for progress bars and file I/O |

> Check `requirements.txt` for exact versions.

---

## 🚀 Installation Guide

### 1️⃣ Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
2️⃣ Upgrade pip
bash
Copy code
python -m pip install --upgrade pip
3️⃣ Install TensorFlow (core dependency)
bash
Copy code
pip install tensorflow-cpu==2.13.0 --no-cache-dir
4️⃣ Install ISR and related packages
bash
Copy code
pip install ISR==2.2.0 --no-deps --no-cache-dir
pip install imageio tqdm pyaml --no-cache-dir
5️⃣ Install all remaining dependencies
bash
Copy code
pip install -r requirements.txt --no-cache-dir
🗂️ Project Structure
bash
Copy code
📦 SRGAN/
 ┣ 📜 app.py                 # Streamlit web application
 ┣ 📜 inference.py           # Prediction/inference script
 ┣ 📓 srgan_training.ipynb   # Training notebook for RRDN/RDN
 ┣ 📜 requirements.txt       # Python dependencies
 ┣ 📜 README.md              # Project documentation
 ┣ 🖼️ demo.gif               # Demo animation
 ┗ 📂 datasets/              # Optional folder for training data
💻 Usage
▶️ Run the Web Application
bash
Copy code
streamlit run app.py
Access the app at http://localhost:8501

🧠 Train the Model
Open srgan_training.ipynb in Jupyter Notebook

Set dataset paths (e.g., E:\datasets\DIV2K) in Cell 5

Run all cells sequentially to train your SRGAN model

🔍 Run Inference
bash
Copy code
python inference.py
✨ Features
✅ Modern Web UI – Built with Streamlit
✅ Image Statistics – View input/output dimensions
✅ Download Option – Save enhanced images
✅ Model Caching – Loads model once for faster response
✅ Real-Time Results – Enhanced output in seconds

🔧 Technical Details
Parameter	Description
Model	RRDN (Recursive Residual Dense Network)
Weights	Pre-trained GAN weights
Upscale Factor	2× or 4× (configurable)
Framework	TensorFlow 2.13.0
Frontend	Streamlit
Backend	Python-based ISR module