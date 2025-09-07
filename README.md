CARETS: A Multi-Task Framework for Time Series Forecasting

📖 Introduction

This repository contains the official implementation of the paper:
"CARETS: A MULTI-TASK FRAMEWORK UNIFYING CLASSIFICATION AND REGRESSION FOR TIME SERIES FORECASTING".

We propose CaReTS, a novel multi-task learning framework for multi-step time series forecasting. The framework jointly models classification and regression tasks through a dual-stream architecture:

Classification branch: captures step-wise future trends.

Regression branch: estimates deviations from the latest target observation.

This design disentangles macro-level trends and micro-level deviations, resulting in more interpretable predictions. To achieve effective joint learning, we introduce a multi-task loss with uncertainty-aware weighting, balancing the contributions of different tasks adaptively.

Four model variants (CaReTS1–4) are developed under this framework, supporting mainstream temporal modeling encoders: CNNs, LSTMs, and Transformers.
Experiments on multiple real-world datasets demonstrate that CaReTS outperforms SOTA algorithms in forecasting accuracy and trend classification.

📊 Datasets

To ensure reproducibility, this repository provides preprocessed datasets used in the paper.

Original datasets can be found here:
👉 [District Microgrid Dataset](https://github.com/FLYao123/District)

Default dataset: unmet power

To switch dataset:
Modify in load_data_new method:

"unmets_HWM_test.csv" → "prs_HWM_test.csv"


Default input/output length: 15-input, 6-output

To change input-output ratio: adjust settings in load_data_new.

💻 Code Structure

This repository provides:

CaReTS1–4 (our proposed models)

Baseline1–3 (newly designed baselines)

Each model has its own folder. Please update absolute paths in the scripts before running.

Code outputs include:

10-fold cross-validation results

Performance on train/validation/test sets

Evaluation metrics: MSE, RMSE, trend accuracy, and average runtime per fold

⚙️ Usage
1. Run with Default LSTM Encoder

By default, the encoder is LSTM.
To switch to CNN or Transformer, modify:

class RegressionDualBranchModel(nn.Module):
    ...
    model_type = "LSTM"  # Change to "CNN" or "TRANSFORMER"

2. Dataset & Input/Output Setup

Default: unmet power dataset, 15-input, 6-output.

To modify:

Dataset: change filename in load_data_new.

Input/output length: adjust parameters in load_data_new.

🔗 References & Acknowledgements

We thank the following contributors for making their codes available:

SOTA baselines (8 models): wuhaixu2016 👉 https://github.com/thuml/Time-Series-Library

(Autoformer, FEDformer, Non-stationary, TimesNet, Dlinear, Nlinear, TimeXer, TimeMixer)

SOTA baselines (2 models): FLYao123 👉 https://github.com/FLYao123/A_Self-organizing_Interval_Type-2_Fuzzy_Neural_Network

(SOIT2FNN-MO, D-CNN-LSTM)

We also express our sincere appreciation to the authors of these 10 algorithms. 🙏

📢 Notes

During the double-blind peer-review stage, the authors will not provide contact information or reply to issues related to this code.

After official acceptance, we will update with personal contact details for academic discussion.
