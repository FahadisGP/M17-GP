# 🗑️ Smart Trash Bin System  

## 📌 Project Overview  
The **Smart Trash Bin System** is an AI-powered waste management solution that detects and sorts plastic water bottles using a ResNet50-based image classification model. Built using a **Raspberry Pi 5**, the system captures images, classifies objects, and rotates the item into the correct bin automatically.  

## 🛠️ Features  
✅ **Automatic Object Detection** – Uses a PIR motion sensor to detect waste.  
✅ **AI-Powered Classification** – Utilizes a pre-trained **ResNet50** model to classify plastic bottles.  
✅ **Motorized Sorting** – Rotates the waste item into the correct bin.  
✅ **Real-time Processing** – Captures and classifies images instantly.  

---

## 🏗️ System Requirements  

### **🔧 Hardware Components**  
- Raspberry Pi 5  
- PIR Motion Sensor  
- Raspberry Pi Camera Module  
- L298N Motor Driver  
- Servo Motor  
- Power Supply  

### **💻 Software & Dependencies**  
- Raspberry Pi OS  
- Python 3.11  
- PyTorch  
- OpenCV  
- lgpio (for GPIO control)  
- NumPy  

---

## 🚀 Installation & Setup  

### **1️⃣ Set Up Raspberry Pi**  
Ensure your Raspberry Pi is running **Raspberry Pi OS**.  

### **2️⃣ Install Dependencies**  
Run the following command to install the necessary packages:  
```bash
pip install torch torchvision opencv-python numpy lgpio
```
### 3️⃣ Install Dependencies
Run the following command to install the necessary Python packages:
```bash
pip install torch torchvision opencv-python numpy lgpio
```

### 4️⃣ Run the System 
Execute the main script to start the smart trash bin system:
```bash
python main.py
```

#### 🧪 Testing
The system has undergone multiple levels of testing, including:

- Sensor Testing – Ensuring PIR motion sensor detects objects accurately.
- Image Classification Testing – Verifying ResNet50 performance on plastic water bottles.
- Motor Control Testing – Ensuring servo motor rotates correctly based on classification.
- End-to-End System Testing – Validating complete functionality under different conditions.

### 📂 Project Structure 
```bash
|-- classify_image.py    # Main script for motion detection, image classification, and sorting
|-- ResNet50_v2.ipynb    # Jupyter Notebook for training the ResNet50 model
|-- resnet_model.pth     # Trained model weights
|-- dataset/             # Folder containing training images
```

### 📈 Future Improvements
- Expand classification to multiple waste categories (paper, metal, glass, etc.).
- Improve real-time processing with edge computing optimizations.
- Integrate IoT for remote monitoring and data analysis.
- Enhance environmental robustness to work in different lighting conditions.

