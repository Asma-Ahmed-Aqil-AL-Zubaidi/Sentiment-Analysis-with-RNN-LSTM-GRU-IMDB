# Sentiment Analysis on IMDb Reviews with RNN, LSTM, and GRU

This repository presents a complete deep learning pipeline for sentiment classification on the IMDb dataset using RNN, LSTM, and GRU architectures, with hyperparameter tuning via Optuna. The project includes data preprocessing, model training, evaluation, and visualization. Built with TensorFlow, Keras, and Scikit-learn.

---

##  Project Highlights

- Preprocessing: cleaning, emoji handling, lemmatization  
- Text vectorization with Keras Tokenizer & padding  
- Deep learning models: SimpleRNN, LSTM, GRU (with Bidirectional layers)  
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- Hyperparameter optimization using Optuna (on GRU)  
- Clear visualizations for metrics & learning curves  
- Trained models saved in `.h5` format  

---

##  Model Architectures

Each model includes:
- Embedding layer  
- Recurrent layers: Bidirectional RNN / LSTM / GRU  
- Dropout + Dense + LeakyReLU  
- Sigmoid output for binary classification  

---

##  Repository Structure


Sentiment-Analysis-IMDb/
├── data/             # IMDb dataset (if customized)
├── preprocessing/    # Cleaning, emoji replacement, lemmatization
├── models/           # RNN, LSTM, GRU implementations
├── optimization/     # Optuna tuning scripts
├── evaluation/       # Metric plots, confusion matrix, reports
├── saved_models/     # .h5 model files
├── notebooks/
│   └── DL3_sentiment_analysis.ipynb
├── requirements.txt
└── README.md





---

##  Performance Summary

| Model  | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| RNN    | ~82.6%   | ~82.3%    | ~83.1% | ~82.7%   |
| LSTM   | ~88.0%   | ~87.3%    | ~88.9% | ~88.1%   |
| GRU    | **~89.8%** | **~89.6%** | **~90.2%** | **~89.9%** |

---

##  Visualizations

- Accuracy & loss curves per model  
- High-confidence predictions  
- Confusion matrices for each classifier  
- Comparative plots (RNN vs LSTM vs GRU)  

---

##  Optuna Hyperparameter Optimization

The GRU model was fine-tuned using Optuna:
- Embedding dimensions: 32, 64, 128  
- GRU units: range 32–128  
- Dropout rates: 0.2 to 0.5  

---

##  Run Instructions

```bash
git clone https://github.com/your-username/Sentiment-Analysis-with-RNN-LSTM-GRU-IMDB.git
cd Sentiment-Analysis-with-RNN-LSTM-GRU-IMDB
pip install -r requirements.txt
python models/gru_model.py
