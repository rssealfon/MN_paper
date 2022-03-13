#Plot ROC curves for high and low proteinuria subsets for ERCB data
###Note: expression data available in GEO, cohort labels available in Table S1

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

ERCB_clinical = pd.read_csv("/Genomics/function/pentacon/sealfon/patient_specific/NEPTUNE/190908_clinical_table/ERCB_glom_RaRachelSealfon .txt", sep="\t")

ERCB_clinical = ERCB_clinical.replace(regex="gg", value="")
ERCB_clinical = ERCB_clinical.replace(regex=" ", value="")
ERCB_clinical = ERCB_clinical.replace(regex=">", value="")
ERCB_clinical = ERCB_clinical.replace(regex="\(NA\?\)", value="")

isHighProt = [ERCB_clinical.Proteinuria[i] != "ND" and ERCB_clinical.Proteinuria[i] != "-" and
 ERCB_clinical.Proteinuria[i] != "micro" and 
 float(ERCB_clinical.Proteinuria[i]) > 3.5 for i in range(len(ERCB_clinical.NewPatientID))]

with open("/Genomics/ogtr04/sealfon/patient_specific/ERCB/ERCB_GlomcustCDF19_new_pat_ID.txt") as textFile: 
 data_cor = [line.rstrip("\n").split("\t") for line in textFile]

tmp = np.transpose(data_cor)
tmp2 = np.delete(tmp, (0,1), axis=0)
dis = [i.split("_")[0] for i in tmp2[:,0]]
id = [i for i in tmp2[:,0]]
tmp2 = np.delete(tmp2, (0), axis=1)
data = [[float(y) for y in x] for x in tmp2]
dis2 = [i=='MGN' for i in dis]

tmp_dis = [(dis[i]=='MGN' or dis[i] =='IgA' or dis[i] =='MCD' or dis[i] =='FSGS') and isHighProt[i] for i in range(len(dis))]

from itertools import compress
dis3 = list(compress(dis, tmp_dis))
dis4 = [i=='MGN' for i in dis3]
data2 = list(compress(data, tmp_dis))

forest = RandomForestClassifier(n_estimators=500, max_features=1000)
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

dis4 = [i=='MCD' for i in dis3]
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


plt.figure(figsize=(7,7))
plt.plot(mean_fpr_mn, mean_tpr_mn, color='b',
         label=r'ERCB MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn, std_auc_mn),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs, mean_tpr_fsgs, color='g',
         label=r'ERCB FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs, std_auc_fsgs),
         lw=2, alpha=.8)
plt.plot(mean_fpr_mcd, mean_tpr_mcd, color='y',
         label=r'ERCB MCD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mcd, std_auc_mcd),
         lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig("ERCB_highupcr.pdf")

###Plot ROC curves for low-proteinuria ERCB participants
tmp = np.transpose(data_cor)
tmp2 = np.delete(tmp, (0,1), axis=0)
dis = [i.split("_")[0] for i in tmp2[:,0]]
id = [i for i in tmp2[:,0]]
tmp2 = np.delete(tmp2, (0), axis=1)
data = [[float(y) for y in x] for x in tmp2]
dis2 = [i=='MGN' for i in dis]

tmp_dis = [(dis[i]=='MGN' or dis[i] =='IgA' or dis[i] =='MCD' or dis[i] =='FSGS') and not isHighProt[i] for i in range(len(dis))]

from itertools import compress
dis3 = list(compress(dis, tmp_dis))
dis4 = [i=='MGN' for i in dis3]
data2 = list(compress(data, tmp_dis))

forest = RandomForestClassifier(n_estimators=500, max_features=1000)
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

plt.figure(figsize=(7,7))
plt.plot(mean_fpr_mn, mean_tpr_mn, color='b',
         label=r'ERCB MN (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mn, std_auc_mn),
         lw=2, alpha=.8)
plt.plot(mean_fpr_fsgs, mean_tpr_fsgs, color='g',
         label=r'ERCB FSGS (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fsgs, std_auc_fsgs),
         lw=2, alpha=.8)
plt.plot(mean_fpr_iga, mean_tpr_iga, color='r',
         label=r'NEPTUNE IgA (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_iga, std_auc_iga),
         lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig("ERCB_lowupcr.pdf")
