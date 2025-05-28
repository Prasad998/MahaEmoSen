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

## 🧩 Problem Formulation

Given a dataset of multimodal Marathi tweets:

\[
\mathcal{D} = \{ T_i \}_{i=1}^{N_t}
\]

where each tweet \( T_i \) is associated with:

- a **textual feature**: \( T_i^t \)  
- an **image feature**: \( I_i \)  
- an **emotion feature**: \( E_i \in \mathcal{E} \)

We define:

\[
\mathcal{E} = \{\text{anger}, \text{fear}, \text{joy}, \text{love}, \text{sadness}, \text{surprise}\}
\]

The task is to predict a **sentiment label** \( S_i \in \mathcal{S} \), where:

\[
\mathcal{S} = \{0\ (\text{negative}),\ 1\ (\text{neutral}),\ 2\ (\text{positive})\}
\]

Thus, the objective is to learn a function:

\[
f: (T_i^t, I_i, E_i) \longrightarrow S_i
\]

such that the sentiment classification is accurate across the multimodal and emotion-embedded inputs.

This is framed as a **multi-class classification** problem. To address the low-resource nature of the Marathi language, we construct a **synthetically augmented dataset** using techniques like back-translation and data diversification. These improvements help our deep learning model generalize better across diverse and context-rich tweet samples.


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
  
**NOTE** -- The Image Feature Files are uploaded in Compressed Format as **>100 MB**.  

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
