####Find AUCs for WCGNA eigengenes (glom data)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd


with open("glom_wgcna2.mes") as textFile: 
 data_cor_neptune = [line.rstrip("\n").split(",") for line in textFile]
with open("arff_files/feature_PAT_Cohort.tmp") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]

forest = RandomForestClassifier(n_estimators=500, max_features=72, class_weight='balanced')

dis4 = [i=='MN' for i in target]
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mn_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_neptune), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_neptune)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_neptune)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mn_n = np.mean(aucs)
std_auc_mn_n = np.std(aucs)
mean_tpr_mn_n = np.mean(tprs, axis=0)

dis4 = [i=='IgA' for i in target]
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_iga_n = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data_cor_neptune), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_neptune)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_neptune)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_iga_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_iga_n = np.mean(aucs)
std_auc_iga_n = np.std(aucs)
mean_tpr_iga_n = np.mean(tprs, axis=0)


dis4 = [i=='MCD' for i in target]
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mcd_n = np.linspace(0, 1, 100)

 i = 0
 for train, test in cv.split(np.array(data_cor_neptune), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_neptune)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_neptune)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mcd_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mcd_n = np.mean(aucs)
std_auc_mcd_n = np.std(aucs)
mean_tpr_mcd_n = np.mean(tprs, axis=0)


dis4 = [i=='FSGS' for i in target]
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_fsgs_n = np.linspace(0, 1, 100)

 i = 0
 for train, test in cv.split(np.array(data_cor_neptune), np.array(dis4)):
  probas_ = forest.fit(np.array(data_cor_neptune)[train], np.array(dis4)[train]).predict_proba(np.array(data_cor_neptune)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_fsgs_n, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_fsgs_n = np.mean(aucs)
std_auc_fsgs_n = np.std(aucs)
mean_tpr_fsgs_n = np.mean(tprs, axis=0)
