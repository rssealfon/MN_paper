####Plot ROC curves for disease classifiers.
###Note: expression data available in GEO, cohort labels available in Table S1

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd

####Construct classifiers for ERCB dataset
with open("/Genomics/ogtr04/sealfon/patient_specific/ERCB/ERCB_GlomcustCDF19_new_pat_ID.txt") as textFile: 
 data_cor = [line.rstrip("\n").split("\t") for line in textFile]

tmp = np.transpose(data_cor)
tmp2 = np.delete(tmp, (0,1), axis=0)
dis = [i.split("_")[0] for i in tmp2[:,0]] 
tmp2 = np.delete(tmp2, (0), axis=1)
data = [[float(y) for y in x] for x in tmp2]
dis2 = [i=='MGN' for i in dis]

tmp_dis = [i=='MGN' or i=='IgA' or i=='MCD' or i=='FSGS' for i in dis]

from itertools import compress
dis3 = list(compress(dis, tmp_dis))
dis4 = [i=='MGN' for i in dis3]
data2 = list(compress(data, tmp_dis))


forest = RandomForestClassifier(n_estimators=500, max_features=1000, class_weight='balanced')
y_pred = cross_val_predict(forest, data2,dis4, cv=5)


dis4 = [i=='MGN' for i in dis3]
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mn = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data2), np.array(dis4)):
  probas_ = forest.fit(np.array(data2)[train], np.array(dis4)[train]).predict_proba(np.array(data2)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mn, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mn = np.mean(aucs)
std_auc_mn = np.std(aucs)
mean_tpr_mn = np.mean(tprs, axis=0)
#plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8)


dis4 = [i=='FSGS' for i in dis3]
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_fsgs = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data2), np.array(dis4)):
  probas_ = forest.fit(np.array(data2)[train], np.array(dis4)[train]).predict_proba(np.array(data2)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_fsgs, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_fsgs = np.mean(aucs)
std_auc_fsgs = np.std(aucs)
mean_tpr_fsgs = np.mean(tprs, axis=0)

dis4 = [i=='IgA' for i in dis3]
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
tprs = []
fprs = []
aucs = []
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_iga = np.linspace(0, 1, 100)
 i = 0
 for train, test in cv.split(np.array(data2), np.array(dis4)):
  probas_ = forest.fit(np.array(data2)[train], np.array(dis4)[train]).predict_proba(np.array(data2)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_iga, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_iga = np.mean(aucs)
std_auc_iga = np.std(aucs)
mean_tpr_iga = np.mean(tprs, axis=0)


dis4 = [i=='MCD' for i in dis3]
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
tprs = []
fprs = []
aucs = []
    
for i in range(0, 10):
 cv = StratifiedKFold(n_splits=5, shuffle=True)
 mean_fpr_mcd = np.linspace(0, 1, 100)

 i = 0
 for train, test in cv.split(np.array(data2), np.array(dis4)):
  probas_ = forest.fit(np.array(data2)[train], np.array(dis4)[train]).predict_proba(np.array(data2)[test])
  fpr, tpr, thresholds = roc_curve(np.array(dis4)[test], probas_[:, 1])
  tprs.append(interp(mean_fpr_mcd, fpr, tpr))
  tprs[-1][0] = 0.0
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  i += 1
mean_auc_mcd = np.mean(aucs)
std_auc_mcd = np.std(aucs)
mean_tpr_mcd = np.mean(tprs, axis=0)

####Construct classifiers for NEPTUNE dataset

with open("../glom_expr_filtered.csv") as textFile: 
 data_cor_neptune = [line.rstrip("\n").split(",") for line in textFile]
with open("../arff_files/feature_PAT_Cohort.tmp") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]


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
#plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8)


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
#plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),


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
#plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

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
#plt.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

####Plot ROC curves

plt.figure(figsize=(7,7))
plt.plot(mean_fpr_mn_n, mean_tpr_mn_n, color='b',
         label=r'NEPTUNE MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn_n, std_auc_mn_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs_n, mean_tpr_fsgs_n, color='g',
         label=r'NEPTUNE FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs_n, std_auc_fsgs_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_iga_n, mean_tpr_iga_n, color='r',
         label=r'NEPTUNE IgA (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_iga_n, std_auc_iga_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mcd_n, mean_tpr_mcd_n, color='y',
         label=r'NEPTUNE MCD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd_n, std_auc_mcd_n),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mn, mean_tpr_mn, color='b',linestyle='dashed',
         label=r'ERCB MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn, std_auc_mn),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs, mean_tpr_fsgs, color='g',linestyle='dashed',
         label=r'ERCB FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs, std_auc_fsgs),
         lw=2, alpha=.8)
plt.plot(mean_fpr_iga, mean_tpr_iga, color='r',linestyle='dashed',
         label=r'ERCB IgA (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_iga, std_auc_iga),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mcd, mean_tpr_mcd, color='y',linestyle='dashed',
         label=r'ERCB MCD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd, std_auc_mcd),
         lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve2.pdf', format='pdf')


