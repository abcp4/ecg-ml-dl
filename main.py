import numpy as np
import pandas as pd
path=''

data=np.load(path+'dataset_processed/data.npy',allow_pickle=True)
Y=np.load(path+'dataset_processed/Y.npy',allow_pickle=True)
labels =pd.read_csv(path+'/dataset_processed/labels.csv')

train_fold=8
val_fold=9
test_fold=10

# 10th fold for testing (9th for now)
X_test = data[labels.strat_fold == test_fold]
y_test = Y[labels.strat_fold == test_fold]
# 9th fold for validation (8th for now)
X_val = data[labels.strat_fold == val_fold]
y_val = Y[labels.strat_fold == val_fold]
# rest for training
X_train = data[labels.strat_fold <= train_fold]
y_train = Y[labels.strat_fold <= train_fold]

from utils import utils
# Preprocess signal data
X_train, X_val, X_test = utils.preprocess_signals(X_train, X_val, X_test,'/content/')
n_classes = y_train.shape[1]


#Treinar modelo


#Avaliar
thresholds=None
tr_df_point = utils.evaluate_experiment( y_train, y_train_pred, thresholds)
print(tr_df_point)
val_df_point = utils.evaluate_experiment( y_val, y_val_pred, thresholds)
print(val_df_point)
