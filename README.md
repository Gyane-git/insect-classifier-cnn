# insect-classifier-cnn
 A deep learning project to classify insect species using Convolutional Neural Networks (CNNs) and Transfer Learning with VGG16. Includes baseline, deeper, and pretrained models with evaluation and performance comparison.


# 🐞 Insect Image Classification using CNNs and Transfer Learning

This project aims to classify insect images into 9 categories using deep learning techniques. It includes a baseline CNN, a deeper CNN architecture, and a transfer learning approach using VGG16. The model is trained and evaluated in Google Colab with image augmentation, metrics reporting, and inference on new images.

## 📂 Dataset

- Located in: `Google Drive > Dataset > Insect Classification`
- Format: Images organized in class-wise folders
- Classes: aphids, beetle, bollworm, bugs, grasshopper, looper, mosquito, sawfly, mites

---

## 🔨 Project Pipeline

### ✅ Step 1: Data Preparation
- Load dataset using `ImageDataGenerator`
- Apply augmentation on training data (rotation, shift, zoom)
- Split into `train`, `val`, and `test` sets
- Handle errors using `safe_generator` to skip unreadable images

### 🧠 Step 2: Baseline CNN
- 3 Convolutional layers with MaxPooling
- 3 Fully connected layers
- Output layer with `softmax` activation
- Optimizer: Adam | Loss: categorical_crossentropy
- Metrics: accuracy
- Plotted loss & accuracy curves

### 🧪 Step 3: Evaluate Baseline Model
- Evaluated on validation set
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix + Classification report
- Observed signs of underfitting

---

### 💪 Step 4: Deeper CNN
- Increased depth (6 Conv layers)
- Added BatchNormalization + Dropout
- Improved accuracy and reduced overfitting

### 📊 Step 5–6: Model Evaluation & Comparison
| Metric       | Baseline CNN | Deeper CNN |
|--------------|--------------|------------|
| Accuracy     | 77%          | 85%        |
| F1-Score     | 0.76         | 0.83       |
| Overfitting  | Low          | Moderate   |

- Compared optimizers: Adam vs SGD
- Trained with Colab GPU (NVIDIA T4)

---

## 🔁 Step 7: Transfer Learning (VGG16)
- Pretrained base: `VGG16 (include_top=False)`
- Custom Head:
  - GlobalAveragePooling
  - Dense(1024, relu)
  - Dense(9, softmax)
- Initial training: Only new head
- Fine-tuning: Unfroze last 4 VGG16 layers with low learning rate

### 📈 Step 8–9: Evaluate Transfer Model
- Highest accuracy achieved: **91%**
- Precision/Recall significantly improved
- Robust generalization, less overfitting

---

## 🔍 Step 10: Inference
- Performed prediction on new/unseen images
- Displayed predicted vs actual classes
- Best performing model: **Transfer Learning (VGG16)**

---

## 📌 Key Learnings
- **Baseline CNN**: Simple, underfit
- **Deeper CNN**: Better, but risk of overfitting
- **Transfer Learning**: Best balance of performance and efficiency
- **Metric Insights**:
  - **Precision**: How often the model was right when it said “apple”
  - **Recall**: How many real “apples” it correctly found
  - **F1**: Balance between both
  - **Support**: Count of test samples per class

---

## 💡 Future Work
- Try other pretrained models: ResNet, Inception
- Collect more diverse insect images
- Implement Grad-CAM for visual explanations
- Convert model for mobile inference (e.g., TensorFlow Lite)

---

## 🧰 Tools Used
- Python, Keras, TensorFlow
- Google Colab (GPU)
- Matplotlib, scikit-learn
- PIL for image handling

---

## 🧠 Author Note
This project is a practical deep learning pipeline designed for educational purposes in AI image classification. It balances theory with hands-on training and testing.

---

