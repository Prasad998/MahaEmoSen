# MahaEmoSen: Emotion-aware Multimodal Framework for Sentiment Analysis in Marathi

---

## 🧠 Overview

**MahaEmoSen** is a novel **multimodal deep learning framework** designed for **sentiment analysis** in **Marathi**, a low-resource language.   
 It leverages:

**- Emotion-aware representation learning from tweets**  
**- Multimodal fusion using textual and visual content**  
**- Word-level attention to filter noisy words**  
**- Data augmentation through back-translation and paraphrasing**  

---

## 📰 Publication

This work has been presented in the paper:  
**["MahaEmoSen: Towards Emotion-aware Multimodal Marathi Sentiment Analysis"](https://doi.org/10.1145/3618057)**

- 🧪 **Status:** _(Published in TALLIP Volume 22, Issue 9)_  [ACM Transactions on Asian and Low-Resource Language Information Processing](https://dl.acm.org/toc/tallip/2023/22/9)
- 🏛️ **Authors:** Prasad Chaudhari, Pankaj Nandeshwar, Shubhi Bansal, Nagendra Kumar
- 🔗 **DOI/Link:** [https://doi.org/10.1145/3618057](https://doi.org/10.1145/3618057)

---

## 📰 Abstract

> With the advent of the Internet, social media platforms have witnessed an enormous increase in user-generated textual and visual content. Microblogs on platforms such as Twitter are extremely useful for comprehending how individuals feel about a specific issue through their posted texts, images, and videos. Owing to the plethora of content generated, it is necessary to derive an insight of its emotional and sentimental inclination. Individuals express themselves in a variety of languages and, lately, the number of people preferring native languages has been consistently increasing. **Marathi language** is predominantly spoken in the Indian state of Maharashtra. However, sentiment analysis in Marathi has rarely been addressed.  
>
> In light of the above, we propose an **emotion-aware multimodal Marathi sentiment analysis method (MahaEmoSen)**. Unlike the existing studies, we leverage **emotions embedded in tweets** besides assimilating the content-based information from the **textual and visual modalities** of social media posts to perform a sentiment classification. We mitigate the problem of small training sets by implementing **data augmentation techniques**. A **word-level attention mechanism** is applied on the textual modality for **contextual inference** and filtering out noisy words from tweets.  
>
> Experimental outcomes on real-world social media datasets demonstrate that our proposed method **outperforms the existing methods** for Marathi sentiment analysis in **resource-constrained circumstances**.

---

---

## 🧪 Key Contributions

- Emotion-tag enhanced multimodal sentiment classifier
- Incorporation of emotional cues improves context sensitivity
- Release of a large-scale annotated Dataset for Sentiment Analysis in Marathi.
- Outperforms strong baselines like **MuRIL**, **MahaRoBERTa**, **IndicBERT**

---

## 🧩 Problem Formulation

Given a dataset of multimodal Marathi tweets:

$$
\mathcal{D} = \{ T_i \}_{i=1}^{N_t}
$$

Each tweet $$\( T_i \)$$ consists of:


$$T_i^t \quad \text{(textual feature)} $$

$$I_i \quad \text{(image feature)} $$

$$E_i \in \mathcal{E} \quad \text{(emotion feature)}$$

The emotion set is defined as:

$$
\mathcal{E} = \{\text{anger}, \text{fear}, \text{joy}, \text{love}, \text{sadness}, \text{surprise}\}
$$

Our goal is to predict the sentiment label $$\( S_i \in \mathcal{S} \),$$ where:

$$
\mathcal{S} = \{0\ (\text{negative}),\ 1\ (\text{neutral}),\ 2\ (\text{positive})\}
$$

The task can be framed as learning a function:

$$
f: (T_i^t, I_i, E_i) \rightarrow S_i
$$

such that the sentiment classification is accurate across the multimodal and emotion-embedded inputs.

This is framed as a **multi-class classification** problem. To address the low-resource nature of the Marathi language, we construct a **synthetically augmented dataset** using techniques like back-translation and data diversification. These improvements help our deep learning model generalize better across diverse and context-rich tweet samples.


---

## 📊 Results Summary (Modality Contribution Study)

| Model Variant         | Precision (%) | Recall (%) | F1-Score (%) | Accuracy (%) |
|-----------------------|---------------|-------------|---------------|---------------|
| MahaEmoSen (T)        | 84.07         | 84.14       | 83.90         | 84.07         |
| MahaEmoSen (I)        | 41.94         | 48.39       | 36.98         | —             |
| MahaEmoSen (T+E)      | 84.67         | 84.68       | 84.65         | 84.67         |
| MahaEmoSen (T+E+I)    | **85.60**     | **85.55**   | **85.57**     | **85.60**     |


### 📊 **Effectiveness Comparison Results on L3CubeMahaSent**

| **Model**             | **Precision (%) - N** | **Precision (%) - Nr** | **Precision (%) - P**  | **Recall (%) - N** | **Recall (%) - Nr**  | **Recall (%) - P**   | **F1-Score (%) - N** | **F1-Score (%) - Nr**  | **F1-Score (%) - P**   | *Avg. Precision (Categorical Accuracy)* |
|-----------------------|------------------------|--------|--------|---------------------|---------|--------|------------------------|---------|--------|---------------------------------------------------|
| BERT-base-uncased     | 70.6                   | 71.8   | 67     | 75.43               | 63.43   | 71.89  | 72.93                  | 67.35   | 69.36  | 69.88                                                |
| Bi-LSTM               | 80                     | 70.2   | 77.2   | 76.92               | 73.89   | 76.44  | 78.43                  | 72      | 76.82  | 75.79                                                |
| CNN                   | 80.6                   | 70     | 77     | 78.39               | 75.27   | 73.89  | 79.48                  | 72.53   | 75.42  | 75.87                                                |
| IndicBERT             | 80.4                   | 69.8   | 76.8   | 78.21               | 75.05   | 73.03  | 79.29                  | 72.33   | 75.22  | 75.67                                                |
| MarathiSentiment      | 85.2                   | 70.4   | 76     | 77.31               | 75.37   | 78.84  | 81.07                  | 72.8    | 77.39  | 77.20                                                |
| MarathiAlBERT         | 89.4                   | 70.8   | 82.6   | 78.7                | 83.89   | 80.98  | 83.71                  | 76.79   | 81.78  | 80.93                                                |
| MahaRoBERTa           | 93                     | 72.2   | 82.2   | 79.22               | 83.18   | 85.8   | 85.56                  | 77.3    | 83.96  | 82.47                                                |
| XLM-RoBERTa-base      | 87                     | 75     | 85     | 86.14               | 80.3    | 80.49  | 86.57                  | 77.56   | 82.68  | 82.33                                                |
| MuRIL                 | 88.8                   | 77.8   | 82     | 85.06               | 79.07   | 84.36  | 86.89                  | 78.43   | 83.16  | 82.67                                                |
| 🏆 **MahaEmoSen**     | **90.6**               | **80** | **86.2** | **88.13**         | **81.63** | **86.9** | **89.35**              | **80.81** | **86.55** | **💯 85.60**                                |

---

## 📂 Code and Datasets

- Original, augmented, and back-translated Marathi tweets
- Emotion annotations and associated images
- Main model code with abalation study codes.
  
> 📢 _**NOTE** -- The Image Feature Files are uploaded in Compressed Format as **>100 MB**_

---

## 🧩 Architecture

<p align="center">
  <img src="Main Architecture.png" alt="MahaEmoSen Architecture" width="500"/>
  <img src="neural nw final.png" alt="Neural Network Layers" width="500"/>
 
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

> 📌 _For rights, collaboration or related access, reach out to the authors._

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 📌 Keywords

> Marathi NLP · Emotion Analysis · Multimodal Sentiment Classification · Social Media · Low-resource Languages · Deep Learning

---
