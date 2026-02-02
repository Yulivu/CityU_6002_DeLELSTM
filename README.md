# DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long term Effects
## Requirements
Python 3.9-3.11 recommended.

PyTorch should be installed separately (CPU or CUDA build depending on your machine).

Other dependencies are listed in requirements.txt (numpy, pandas, matplotlib, torchmetrics, scikit-learn).

All the codes are run on GPUs by default.

## PM2.5 Experiment
The PM2.5 data can be downloaded from https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

## Electricity Experiment
The electricity data can be downloaded from https://www.kaggle.com/datasets/unajtheb/homesteadus-electricity-consumption

## Exchange Experiment 
The exchange data can be downloaded from https://github.com/laiguokun/multivariate-time-series-data

## Training 
The following commands will train three task-specific datasets. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

```
python Electricity.py
python Exchange.py
python PM.py
```

Optional arguments:
- --data_dir: directory containing newX_train.csv and second_y.csv (default uses DATA/<dataset> in this repo)
- --device: torch device string, e.g. cpu or cuda:0 (default auto-select)
- --epochs: number of training epochs
- --models: comma-separated model names, e.g. Delelstm or Delelstm,LSTM
