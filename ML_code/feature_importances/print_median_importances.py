#Print median rank importance of each gene across 50 runs of random forest classifier
###Note: expression data available in GEO, cohort labels available in Table S1

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

with open("glom_expr_filtered.csv") as textFile: 
 X = [line.rstrip("\n").split(",") for line in textFile]
with open("feature_PAT_Cohort.tmp") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]

target2 = [i=='MN' for i in target]

forest = RandomForestClassifier(n_estimators=500, max_features=1000)

importances_v=[]
i = 0
for i in range(0,50):
 forest.fit(X,target2)
 importances_v.append(forest.feature_importances_)
 i += 1

ranks = [len(importances_v[0]) - stats.rankdata(i) for i in importances_v]
median_ranks = np.median(ranks, axis=0)

for i in range(0, len(median_ranks)):
 print(median_ranks[i])
