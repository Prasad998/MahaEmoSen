# MahaEmoSen

**Emotion-aware Multimodal Framework for Sentiment Analysis in Marathi**

---

## 🧠 Overview

**MahaEmoSen** is a novel **multimodal deep learning framework** designed for **sentiment analysis** in **Marathi**, a low-resource language.   
 It leverages:

**- Emotion-aware representation learning from tweets**  
**- Multimodal fusion using textual and visual content**  
**- Word-level attention to filter noisy words**  
**- Data augmentation through back-translation and paraphrasing**  

---

## 📰 Abstract

> With the advent of the Internet, social media platforms have seen an enormous rise in user-generated content. Tweets often contain both text and images and are rich in sentiment and emotion. While sentiment analysis in English and major global languages has progressed rapidly, **Marathi**—a widely spoken Indian language—has seen limited research.
>
> We propose **MahaEmoSen**, an **emotion-aware multimodal sentiment classification model** that integrates **textual**, **visual**, and **emotional** information. We further tackle the scarcity of training data through robust augmentation techniques. A **word-level attention layer** is applied for contextual refinement, while emotion-tags aid sentiment prediction.
>
> Experimental results reveal that **MahaEmoSen outperforms baseline models** significantly in emotion-rich and data-scarce settings.

---

## 🧪 Key Contributions

- Emotion-tag enhanced multimodal sentiment classifier
- Incorporation of emotional cues improves context sensitivity
- Release of a large-scale annotated Dataset for Sentiment Analysis in Marathi.
- Outperforms strong baselines like **MuRIL**, **MahaRoBERTa**, **IndicBERT**

---

## 🧾 Problem Formulation

Given a dataset \( D \) of multimodal tweets \( T_i \) with:

- Textual feature \( T_t \)
- Emotion feature \( E_i \in \{	ext{anger, fear, joy, love, sadness, surprise}\} \)
- Image feature \( I_i \)

The task is to classify sentiment \( S_i \in \{0: 	ext{Negative}, 1: 	ext{Neutral}, 2: 	ext{Positive} \} \).

---

## 📊 Results Summary

| Model Variant         | Precision (%) | Recall (%) | F1-Score (%) | Accuracy (%) |
|-----------------------|---------------|-------------|---------------|---------------|
| MahaEmoSen (T)        | 84.07         | 84.14       | 83.90         | 84.07         |
| MahaEmoSen (I)        | 41.94         | 48.39       | 36.98         | —             |
| MahaEmoSen (T+E)      | 84.67         | 84.68       | 84.65         | 84.67         |
| MahaEmoSen (T+E+I)    | **85.60**     | **85.55**   | **85.57**     | **85.60**     |

### 📈 Baseline Comparison

| Model             | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| BERT-base        | 69.88     | 67.35  | 69.36    |
| CNN              | 75.87     | 72.53  | 75.42    |
| MarathiSentiment | 77.20     | 72.80  | 77.39    |
| MahaRoBERTa      | 82.47     | 77.30  | 83.96    |
| MuRIL            | 82.67     | 78.43  | 83.16    |
| **MahaEmoSen**   | **85.60** | 85.55  | **85.57**|

---

## 📂 Dataset

- Original, augmented, and back-translated Marathi tweets
- Emotion annotations and associated images

> 📢 _Dataset download link coming soon. Contact the authors for early access._

---

## 🧩 Architecture

<p align="center">
  <img src="assets/mahaemosen-architecture.png" alt="MahaEmoSen Architecture" width="600"/>
</p>

---

## ⚙️ Setup Instructions

### Requirements

```bash
pip install -r requirements.txt
```

### Repository Structure

```bash
MahaEmoSen/
├── data/                 # Datasets and annotations
├── models/               # Pretrained models and checkpoints
├── src/                  # Source code (model, training, utils)
├── assets/               # Images and diagrams
├── requirements.txt
└── main.py               # Entry point script
```

---

## 🚀 Usage Example

```python
from src.model import MahaEmoSenModel
from src.utils import load_data

# Load preprocessed data
text, image, emotion = load_data('data/test_sample.json')

# Initialize model
model = MahaEmoSenModel()

# Predict sentiment
pred = model.predict(text_input=text, image_input=image, emotion_input=emotion)
print("Predicted sentiment:", pred)
```

---

## 📚 Citation

> 📌 Citation will be added after formal publication.  
> _For preprint, collaboration or early access, reach out to the authors._

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 📌 Keywords

> Marathi NLP · Emotion Analysis · Multimodal Sentiment Classification · Social Media · Low-resource Languages · Deep Learning

---
