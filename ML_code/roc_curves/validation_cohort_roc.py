#Plot ROC curves for disease classifiers for validation cohort
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

with open("validation_expr.txt") as textFile: 
    data_cor_expanded = [line.rstrip("\n").split("\t") for line in textFile]
data_cor_expanded = data_cor_expanded[1:len(data_cor_expanded)]
with open("validation_cohort_IgA.txt") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]
data_cor_expanded = np.transpose(data_cor_expanded)

dis4 = [i=='1 - MN' for i in target]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mn_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_expanded), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_expanded)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_expanded)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mn_n = np.mean(aucs)
std_auc_mn_n = np.std(aucs)
mean_tpr_mn_n = np.mean(tprs, axis=0)

dis4 = [i=='2 - MCD' for i in target]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mcd_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_expanded), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_expanded)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_expanded)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mcd_n = np.mean(aucs)
std_auc_mcd_n = np.std(aucs)
mean_tpr_mcd_n = np.mean(tprs, axis=0)

dis4 = [i=='4 - FSGS' for i in target]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_fsgs_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_expanded), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_expanded)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_expanded)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_fsgs_n = np.mean(aucs)
std_auc_fsgs_n = np.std(aucs)
mean_tpr_fsgs_n = np.mean(tprs, axis=0)

dis4 = [i=='IgA' for i in target]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_iga_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_expanded), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_expanded)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_expanded)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_iga_n = np.mean(aucs)
std_auc_iga_n = np.std(aucs)
mean_tpr_iga_n = np.mean(tprs, axis=0)

plt.figure(figsize=(7,7))
plt.plot(mean_fpr_mn_n, mean_tpr_mn_n, color='b',
         label=r'MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn_n, std_auc_mn_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs_n, mean_tpr_fsgs_n, color='g',
         label=r'FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs_n, std_auc_fsgs_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_iga_n, mean_tpr_iga_n, color='r',
         label=r'NEPTUNE IgA (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_iga_n, std_auc_iga_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mcd_n, mean_tpr_mcd_n, color='y',
         label=r'MCD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd_n, std_auc_mcd_n),
         lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.savefig('validation_roc_curve_iga.pdf', format='pdf')