# 🐾 AnimalVision

AnimalVision is a deep learning project that uses a Convolutional Neural Network (CNN) to classify animal images. With a dataset of thousands of images and 90 different animal classes, the model achieves high accuracy in predictions.

---

## 📌 Project Overview

- **Model**: `AnimalVisionModel.keras`
- **Dataset**: 5,400 images across 90 animal classes
- **Architecture**: CNN-based classification model
- **Performance Metrics**:
  - **Validation Accuracy**: 85%
  - **AUC Score**: 99%

This model has been trained to recognize a diverse set of animals with high precision.

---

## 🚀 Try It Out!

You can test the model using our **Streamlit web demo**:  
🔗 [AnimalVision Demo](https://animalvision.streamlit.app)

### How to Use the Demo:
1. Upload an image of an animal.
2. The model will analyze the image and return the **top predicted class**.
3. A **bar chart** will display the top 5 predictions with their confidence scores.

---

## 🛠 Installation & Usage

If you want to run the project locally, follow these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/furkankarakuz/AnimalVision.git
cd AnimalVision
```

### 2️⃣ Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

Now you can access the app locally and test images directly from your browser!

---

## 📦 Model Details

The **AnimalVisionModel** was trained on a dataset of **5,400 images** across **90 animal categories**. The training process involved:
- Data augmentation techniques to improve generalization.
- A CNN architecture optimized for image classification.
- Evaluation using **accuracy** and **AUC** metrics.

With a validation accuracy of **85%** and an AUC score of **99%**, the model demonstrates strong classification capabilities.

---

## 📊 Example Predictions
Below is an example output when testing an image:

| Rank | Animal | Confidence |
|------|--------|------------|
| 1️⃣  | Tiger 🐯 | 92% |
| 2️⃣  | Lion 🦁 | 87% |
| 3️⃣  | Cheetah 🐆 | 79% |
| 4️⃣  | Leopard 🐆 | 73% |
| 5️⃣  | Jaguar 🐆 | 69% |

This example demonstrates how the model ranks the possible classifications with confidence scores.

---

## 📢 Contributing

Contributions are welcome! If you have ideas for improvement or want to enhance the model, feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

🐾 **Explore the world of animals with AI!** 🎉
