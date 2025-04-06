# Analysing Bias in Real vs Fake News Classification using LSTM

This project investigates the vulnerability of Long Short-Term Memory (LSTM) models to spurious correlations in the context of fake news classification. Through systematic manipulation of the dataset, we reveal how biases in text data can significantly inflate model performance, ultimately misleading the evaluation of deep learning models in NLP tasks.

## Background

Fake news detection is a challenging natural language processing (NLP) task that relies on understanding subtle linguistic cues. However, models like LSTM can inadvertently exploit superficial features or artifacts in the data (e.g., stylistic differences between real and fake news) that do not generalise well to real-world scenarios.

This project tests the hypothesis that LSTMs are prone to such spurious correlations and evaluates their performance under more controlled, bias-mitigated conditions.

## Methodology

To assess LSTM vulnerability to spurious correlations in fake news classification, the following approach was taken:

- **Dataset**: A balanced REAL/FAKE news dataset from Kaggle (6,335 samples), split into training (72%), validation (18%), and test (10%) sets.
  
- **Spurious Correlation Injection**:
  - Two meaningless tokens (`jugfsd`, `thadfj`) were injected into the **training and validation sets**, with strong (but imperfect) correlation to class labels.
  - The **test set** was manipulated with the inverse correlation to expose model reliance on these tokens.

- **Model**:  
  - Custom LSTM classifier with an embedding layer (trained from scratch), an LSTM layer, and a fully connected layer with sigmoid activation.
  - Pre-trained embeddings were avoided to allow learning of the injected tokens.

- **Evaluation**:
  - Performance compared using accuracy and F1-score on clean vs manipulated datasets.
  - Grad-CAM was used to interpret model behaviour and verify reliance on spurious features.

## Results

The LSTM model showed a significant performance drop when tested on the manipulated dataset, indicating that it had learned to rely on the artificial spurious tokens introduced during training.

Grad-CAM visualizations further confirmed that these tokens contributed heavily to the model’s decision-making, despite their lack of semantic meaning.

## Technologies

- Python
- PyTorch
- Grad-CAM
- Numpy, Pandas, Matplotlib
