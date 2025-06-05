GitHub Repository for CS231N Final Project: Cap4Art: Improving Image Captioning Capabilities Through Multi-Task Learning

# code

The emotion folder contains code used for training classifiers based on emotion label datasets.

semart
fine-tunes the convnext model for school, time frame, and type classifications.

---

## 📄 Contents

- inference.py: used to generate classifications using the model checkpoint.
  style.py: used for training the classifier to predict styles from the Artemis Dataset.
  train_distribution.py: predict the emotion labels in WikiArt Emotion based on KL divergence loss with the human labels.

---
