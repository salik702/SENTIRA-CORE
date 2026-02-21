## 🌐 Live Demo

[🚀 Launch the App](https://sentira-core.streamlit.app/) *(Placeholder Link)*

## 🖼️ Preview

<div align="center">
  <img width="872" alt="Sentira Core Preview" src="https://github.com/user-attachments/assets/placeholder-image-id" />
</div>

> *A stunning, glassmorphism-inspired dark UI that makes sentiment analysis a visual delight.*

---

## 🚀 Key Features

### 🔮 **Neural Sentiment Processing**

- Leverages a **Linear Support Vector Classifier (SVC)** to analyze textual patterns with high fidelity.
- Processes input streams in real-time to extract emotional signatures.

### 🌈 **Emotional Spectrum Index**

- Classifies text into six distinct emotional states:
  - **Positive**: Joy, Love, Surprise
  - **Negative**: Sadness, Anger, Fear
- Provides a "Neural Resonance" score (Confidence) for every analysis.

### 🎨 **Ultra-Modern UI/UX**

- **Glassmorphism Design**: Translucent containers with background blur effects and neon accents.
- **Dynamic Animations**: Fluid "reveal" animations and pulsing UI elements.
- **Aggressive Clean UI**: Completely hidden Streamlit headers for a standalone app feel.

---

## 🛠️ Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-46AF8C?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

---

## 📦 Installation (Local Setup)

Get the engine running on your local machine in minutes.

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd sentira-core
```

**2. Set up virtual environment (Recommended)**

*Windows:*

```bash
python -m venv venv
.\venv\Scripts\activate
```

*macOS/Linux:*

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install streamlit joblib nltk scikit-learn scipy pandas
```

**4. Launch the App**

```bash
streamlit run app.py
```

---

## 📈 Model Performance

- **Algorithm:** Linear Support Vector Classifier (LinearSVC) 💠
- **Performance Metrics:**
  - **Accuracy Score:** `~89.2%`
  - **Preprocessing:** TF-IDF Vectorization with custom NLTK stopword filtering.
- **Key Emotions:** Joy, Love, Surprise, Sadness, Anger, Fear.

---

## 💡 How It Works

1. **Input Stream**: The system accepts raw text through the glassmorphism input portal.
2. **Preprocessing**: Text is normalized (lower-cased), punctuation/numbers are removed, and non-ASCII characters are filtered.
3. **Vectorization**: The processed text is transformed using a pre-trained **TF-IDF Vectorizer**.
4. **Neural Analysis**: The **Linear SVC model** calculates decision function scores for each emotional class.
5. **Softmax Calibration**: Scores are passed through a softmax function to generate confidence percentages.
6. **Visual Output**: The primary emotion and confidence score are rendered in a high-contrast result card.

---

## 🤝 Contribution

We welcome contributions! Whether it's enhancing the model, improving the UI, or adding new emotional classes:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open-source and available for free.

---

<div align="center">
  <p>Made with ❤️ and 💠 by <b>Salik Ahmad</b></p>
  <p>
    <a href="https://salikahmad.vercel.app/">🌐 Portfolio</a>  |  
    <a href="https://www.linkedin.com/in/salik-ahmad-6b0142273/">🔗 LinkedIn</a>
  </p>
</div>
