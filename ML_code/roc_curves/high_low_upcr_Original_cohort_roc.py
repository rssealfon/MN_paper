#Plot ROC curves for high and low-proteinuria subsets of Original cohort
###Note: expression data available in GEO, cohort labels available in Table S1

import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np

with open("hi_prot_expr_orig.txt") as textFile: 
    data_cor_filtered = [line.rstrip("\n").split("\t") for line in textFile]

with open("hi_prot_expr_cohort_orig.txt") as f: 
 target_filtered = [line.rstrip('\n').rstrip('\r') for line in f]

dis4 = [i=='MN' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mn_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mn_n = np.mean(aucs)
std_auc_mn_n = np.std(aucs)
mean_tpr_mn_n = np.mean(tprs, axis=0)

dis4 = [i=='MCD' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mcd_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mcd_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mcd_n = np.mean(aucs)
std_auc_mcd_n = np.std(aucs)
mean_tpr_mcd_n = np.mean(tprs, axis=0)

dis4 = [i=='FSGS' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_fsgs_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_fsgs_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_fsgs_n = np.mean(aucs)
std_auc_fsgs_n = np.std(aucs)
mean_tpr_fsgs_n = np.mean(tprs, axis=0)

plt.figure(figsize=(7,7))
plt.plot(mean_fpr_mn_n, mean_tpr_mn_n, color='b',
         label=r'NEPTUNE MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn_n, std_auc_mn_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs_n, mean_tpr_fsgs_n, color='g',
         label=r'NEPTUNE FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs_n, std_auc_fsgs_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mcd_n, mean_tpr_mcd_n, color='y',
         label=r'NEPTUNE MCD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd_n, std_auc_mcd_n),
         lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.savefig('high_upcr_roc_curve_orig.pdf', format='pdf')

with open("lo_prot_expr_orig.txt") as textFile: 
    data_cor_filtered = [line.rstrip("\n").split("\t") for line in textFile]
with open("lo_prot_expr_cohort_orig.txt") as f: 
 target_filtered = [line.rstrip('\n').rstrip('\r') for line in f]   

dis4 = [i=='MN' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mn_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mn_n = np.mean(aucs)
std_auc_mn_n = np.std(aucs)
mean_tpr_mn_n = np.mean(tprs, axis=0)

dis4 = [i=='MCD' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mcd_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mcd_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mcd_n = np.mean(aucs)
std_auc_mcd_n = np.std(aucs)
mean_tpr_mcd_n = np.mean(tprs, axis=0)

dis4 = [i=='FSGS' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_fsgs_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_fsgs_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_fsgs_n = np.mean(aucs)
std_auc_fsgs_n = np.std(aucs)
mean_tpr_fsgs_n = np.mean(tprs, axis=0)

dis4 = [i=='IgA' for i in target_filtered]
#forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_iga_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_filtered), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_filtered)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_filtered)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_iga_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_iga_n = np.mean(aucs)
std_auc_iga_n = np.std(aucs)
mean_tpr_iga_n = np.mean(tprs, axis=0)

plt.figure(figsize=(7,7))
plt.plot(mean_fpr_mn_n, mean_tpr_mn_n, color='b',
         label=r'NEPTUNE MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn_n, std_auc_mn_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs_n, mean_tpr_fsgs_n, color='g',
         label=r'NEPTUNE FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs_n, std_auc_fsgs_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mcd_n, mean_tpr_mcd_n, color='y',
         label=r'NEPTUNE MCD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd_n, std_auc_mcd_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_iga_n, mean_tpr_iga_n, color='r',
         label=r'NEPTUNE IgA (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd_n, std_auc_mcd_n),
         lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.savefig('lo_upcr_roc_curve_orig.pdf', format='pdf')

