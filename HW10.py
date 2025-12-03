import os
import scipy.io as sio
from self_py_fun.HW10Fun import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In HW7, you have the chance to visualize a truncated EEG dataset stratified by
# target and non-target stimulus type.
#
# The fundamental problem of P300 ERP-BCI speller system is to perform a binary classification.
#
# In HW10, you are asked to implement the binary classification using various methods,
# and evaluate the model performance with a testing dataset.
#
# You will use K114_001_BCI_TRN_Truncated_Data_0.5_6.mat as a training set, and
# K114_001_BCI_FRT_Truncated_Data_0.5_6.mat as a testing set.
#
# Notice that here, we do not split training/testing within K114_001_BCI_TRN_Truncated_Data_0.5_6.mat
# because each row is not entirely independent of each other due to the special structure of the dataset.

# Global constants:
np.random.seed(100)
bp_low = 0.5
bp_upp = 6
electrode_num = 16
# Change the following directory to your own one.
parent_dir = '/Users/sapnadavid/Documents/GitHub/BIOS-584/BIOS-584'
parent_data_dir = '{}/data'.format(parent_dir)
time_index = np.linspace(0, 800, 25)
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
subject_name = 'K114'
# create a new folder called K114
subject_dir = '{}/{}'.format(parent_dir, subject_name)
if not os.path.exists(subject_dir):
    os.mkdir(subject_dir)

char_trn = 'THE0QUICK0BROWN0FOX'
char_trn_size = len(char_trn)

# Step 1: Import dataset
# Step 1.1: TRN dataset
trn_data_name = '{}_001_BCI_TRN_Truncated_Data_{}_{}'.format(subject_name, bp_low, bp_upp)
trn_data_dir = '{}/{}.mat'.format(parent_data_dir, trn_data_name)
eeg_trn_obj = sio.loadmat(trn_data_dir)

import os
print(os.path.exists(trn_data_dir))
print(trn_data_dir)

# eeg_trn_obj is a dictionary!
print(eeg_trn_obj.keys())
eeg_trn_signal = eeg_trn_obj['Signal']
print(eeg_trn_signal.shape) # 3420, 400
eeg_trn_type = eeg_trn_obj['Type']
print(eeg_trn_type.shape) # 3420, 1
eeg_trn_type = np.squeeze(eeg_trn_type, axis=1)

# Step 1.2: FRT dataset
# The following code should be completed by students themselves.
# you should be able to obtain relevant data files named
# eeg_frt_signal and eeg_frt_type
# Write your own code below:
frt_data_name = '{}_001_BCI_FRT_Truncated_Data_{}_{}'.format(subject_name, bp_low, bp_upp)
frt_data_dir = '{}/{}.mat'.format(parent_data_dir, frt_data_name)
eeg_frt_obj = sio.loadmat(frt_data_dir)

# eeg_frt_obj is also a dictionary
print(eeg_frt_obj.keys())

eeg_frt_signal = eeg_frt_obj['Signal']
print(eeg_frt_signal.shape)

eeg_frt_type = eeg_frt_obj['Type']
print(eeg_frt_type.shape)

# squeeze type vector
eeg_frt_type = np.squeeze(eeg_frt_type, axis=1)




# You have completed the exploratory data analysis in HW7 and HW8.
# The dataset has been carefully reviewed by Dr. Jane E. Huggins,
# so we do not need to worry about missing, outliers, errors of the dataset.

# Step 2: Fit classification models
# You will try the following methods:
# Logistic Regression,
# Linear Discriminant Analysis,
# Support Vector Machine (sometimes called support vector classification)
# You do not need to modify the parameters of each classifier
# except for LogisticRegression: set max_iter=1000
# Write your own code below:
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
eeg_trn_signal_scaled = scaler.fit_transform(eeg_trn_signal)
eeg_frt_signal_scaled = scaler.transform(eeg_frt_signal)
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(eeg_trn_signal_scaled, eeg_trn_type)
log_reg_pred = log_reg.predict(eeg_frt_signal_scaled)
log_reg_prob = log_reg.predict_proba(eeg_frt_signal)[:, 1]

# Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis()
lda.fit(eeg_trn_signal, eeg_trn_type)
lda_pred = lda.predict(eeg_frt_signal)
lda_prob = lda.predict_proba(eeg_frt_signal)[:, 1]

# Support Vector Machine (SVC)
svc = SVC(probability=True)
svc.fit(eeg_trn_signal, eeg_trn_type)
svc_pred = svc.predict(eeg_frt_signal)
svc_prob = svc.predict_proba(eeg_frt_signal)[:, 1]




## Step 3.1: Prediction accuracy on TRN files (stimulus-level)

# Logistic Regression: probability of class 1
logistic_y_trn = log_reg.predict_proba(eeg_trn_signal_scaled)[:, 1].reshape(-1, 1)

# LDA: probability of class 1
lda_y_trn = lda.predict_proba(eeg_trn_signal_scaled)[:, 1].reshape(-1, 1)

# SVM: probability of class 1
svm_y_trn = svc.predict_proba(eeg_trn_signal_scaled)[:, 1].reshape(-1, 1)

# Step 3.2: Prediction accuracy on FRT files (stimulus-level)

# Logistic Regression: probability of class 1
logistic_y_frt = log_reg.predict_proba(eeg_frt_signal_scaled)[:, 1].reshape(-1, 1)

# LDA: probability of class 1
lda_y_frt = lda.predict_proba(eeg_frt_signal_scaled)[:, 1].reshape(-1, 1)

# SVM: probability of class 1
svm_y_frt = svc.predict_proba(eeg_frt_signal_scaled)[:, 1].reshape(-1, 1)

# Step 4: Convert stimulus-level probability → character-level accuracy

eeg_trn_code = eeg_trn_obj['Code']
eeg_frt_code = eeg_frt_obj['Code']

char_frt = convert_raw_char_to_alphanumeric_stype(eeg_frt_obj['Text'])
char_frt_size = len(char_frt)
frt_seq_size = int(eeg_frt_signal.shape[0] / char_frt_size / 12)

# Logistic Regression

print('Logistic Regression on TRN:')
logistic_letter_mat_trn, logistic_letter_prob_mat_trn = streamline_predict(
    logistic_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
logistic_trn_accuracy = np.mean(logistic_letter_mat_trn ==
                                np.array(list(char_trn))[:, np.newaxis], axis=0)

print('Logistic Regression on FRT:')
logistic_letter_mat_frt, logistic_letter_prob_mat_frt = streamline_predict(
    logistic_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
logistic_frt_accuracy = np.mean(logistic_letter_mat_frt ==
                                np.array(list(char_frt))[:, np.newaxis], axis=0)

# LDA

print('LDA on TRN:')
lda_letter_mat_trn, lda_letter_prob_mat_trn = streamline_predict(
    lda_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
lda_trn_accuracy = np.mean(lda_letter_mat_trn ==
                           np.array(list(char_trn))[:, np.newaxis], axis=0)

print('LDA on FRT:')
lda_letter_mat_frt, lda_letter_prob_mat_frt = streamline_predict(
    lda_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
lda_frt_accuracy = np.mean(lda_letter_mat_frt ==
                           np.array(list(char_frt))[:, np.newaxis], axis=0)

# SVM
print('Support Vector Machine on TRN:')
svm_letter_mat_trn, svm_letter_prob_mat_trn = streamline_predict(
    svm_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
svm_trn_accuracy = np.mean(svm_letter_mat_trn ==
                           np.array(list(char_trn))[:, np.newaxis], axis=0)

print('Support Vector Machine on FRT:')
svm_letter_mat_frt, svm_letter_prob_mat_frt = streamline_predict(
    svm_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
svm_frt_accuracy = np.mean(svm_letter_mat_frt ==
                           np.array(list(char_frt))[:, np.newaxis], axis=0)

# Print results

print(logistic_trn_accuracy)
print(lda_trn_accuracy)
print(svm_trn_accuracy)

print(logistic_frt_accuracy)
print(lda_frt_accuracy)
print(svm_frt_accuracy)


# Remember to answer two questions below:

# What do rows 122, 131, 141, 150, 160, and 169 do? Briefly answer the question below:
#These lines compare predicted characters to the true characters and compute the model’s accuracy across all
# trials for Logistic Regression (TRN & FRT), LDA (TRN & FRT), and SVM (TRN & FRT). So each line outputs the
# accuracy (%) of one model on one dataset.

logistic_trn_accuracy = np.mean(logistic_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
logistic_frt_accuracy = np.mean(logistic_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)
lda_trn_accuracy = np.mean(lda_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
lda_frt_accuracy = np.mean(lda_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)
svm_trn_accuracy = np.mean(svm_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
svm_frt_accuracy = np.mean(svm_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

# Step 5: Summary
# Which method performs the best? Why?
#LDA performs the best.
#This is because EEG P300 data is approximately linearly separable when averaged, and LDA is optimized
#for finding a linear boundary that maximizes separation between target and non-target responses.
# It is also more stable with high-dimensional,  noisy data and typically outperforms Logistic Regression
# and SVM in traditional P300 BCI tasks.