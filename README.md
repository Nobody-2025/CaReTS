CARETS: A Multi-Task Framework for Time Series Forecasting

📖 Introduction

This repository contains the official implementation of the manuscript (submitted): "CARETS: A MULTI-TASK FRAMEWORK UNIFYING CLASSIFICATION AND REGRESSION FOR TIME SERIES FORECASTING".

We propose CaReTS, a novel multi-task learning framework for multi-step time series forecasting. The framework jointly models classification and regression tasks through a dual-stream architecture:

1) Classification branch: captures step-wise future trends;

2) Regression branch: estimates deviations from the latest target observation.

This design disentangles macro-level trends and micro-level deviations, resulting in more interpretable predictions. To achieve effective joint learning, we introduce a multi-task loss with uncertainty-aware weighting, balancing the contributions of different tasks adaptively. Four model variants (CaReTS1–4) are developed under this framework, supporting mainstream temporal modeling encoders: CNNs, LSTMs, and Transformers. Experiments on multiple real-world datasets demonstrate that CaReTS outperforms SOTA algorithms in forecasting accuracy and trend classification.

📊 Datasets

To ensure reproducibility, this repository provides preprocessed datasets used in the manuscript. 
Original datasets can be found here: 👉  [FLYao123] https://github.com/FLYao123/District-microgrid-dataset


💻 Code Structure

This repository provides: CaReTS1–4 (our proposed models) and Baseline1–3 (newly designed baselines). Each model has its own file. 📢 Please update the absolute paths in the scripts before running.

Code outputs include: 1) N-fold cross-validation results; 2) Performance on train/validation/test sets.

Evaluation metrics: MSE, RMSE, trend accuracy, and average runtime per fold.

⚙️ Usage

1. Encoder Setup

By default, the encoder is LSTM. To switch to CNN or Transformer, e.g., modify:

    Class RegressionDualBranchModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_len=6, model_type='LSTM'):# Change to "CNN" or "TRANSFORMER"
        ...
        
2. Dataset Setup

By default, the dataset is unmet power. To switch to electricity price dataset, modify:

    def load_data_new:
        ...
        X_data_train = pd.read_csv('/content/drive/MyDrive/LSTM_comparison/Chaotic_data/unmets_HWM_train.csv') # Change to "prs_HWM_train.csv"
        X_data_test = pd.read_csv('/content/drive/MyDrive/LSTM_comparison/Chaotic_data/unmets_HWM_test.csv')# Change to "prs_HWM_train.csv"
        ...


3. Dataset & Input/Output Setup

By default, the forecasting mode is 15-input-6-output. To change input-output ratio: adjust settings when loading the training and test sets in load_data_new method. If you modify the output length, please remember to also update the output_len parameter in RegressionDualBranchModel(nn.Module), for example::
    
    Class RegressionDualBranchModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_len=6, model_type='LSTM'):# Change '6' to '4' or '8'
        ...
During training, the model automatically detects the input length, so no further modification is required.

🔗 References & Acknowledgements

We thank the following contributors for making their codes available:

1) SOTA baselines (8 models: Autoformer, FEDformer, Non-stationary, TimesNet, Dlinear, Nlinear, TimeXer, TimeMixer) 👉 [wuhaixu2016]  https://github.com/thuml/Time-Series-Library


2) SOTA baselines (2 models: SOIT2FNN-MO, D-CNN-LSTM) 👉 [FLYao123] https://github.com/FLYao123/A_Self-organizing_Interval_Type-2_Fuzzy_Neural_Network

3) We also express our sincere appreciation to the authors of these 10 algorithms. 🙏

📢 Notes

During the double-blind peer-review stage, the authors will not provide contact information or reply to issues related to this code. After official acceptance, we will update with contact details for academic discussion.
