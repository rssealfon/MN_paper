####Find AUCs of top k features sorted by importance
##Note: expression data available in GEO, cohort labels available in Table S1

from sklearn.ensemble import RandomForestClassifier

#Train on first 2/3 of data
with open("glom_expr_filtered_head.csv") as textFile: 
 data_cor = [line.rstrip("\n").split(",") for line in textFile]

data_cor = [[float(y) for y in x] for x in data_cor]

with open("arff_files/feature_PAT_Cohort_head.glom") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]

for i in range(0,len(target)):
 if target[i]=='MN':
  target[i] = 1
 else:
  target[i] = 0

forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest.fit(data_cor, target)
forest.feature_importances_

#Evaluate on final 1/3 of data
with open("glom_expr_filtered_tail.csv") as textFile: 
 data2 = [line.rstrip("\n").split(",") for line in textFile]

data2 = [[float(y) for y in x] for x in data2]

with open("arff_files/feature_PAT_Cohort_tail.glom") as f: 
 target2 = [line.rstrip('\n').rstrip('\r') for line in f]

for i in range(0,len(target2)):
 if target2[i]=='MN':
  target2[i] = 1
 else:
  target2[i] = 0


import numpy as np
indices = np.argsort(forest.feature_importances_)[::-1]

from sklearn.metrics import roc_auc_score

roc_auc_score(target2, forest.predict_proba(data2)[:,1])

data_cor = np.array(data_cor)
forest = RandomForestClassifier(n_estimators=500, max_features=10)

for i in range(1,501):
 data_red = data_cor[:,indices[0:i]]
 forest = RandomForestClassifier(n_estimators=500, max_features=i)
 fit = forest.fit(data_red, target)
 data2_red = np.array(data2)[:,indices[0:i]]
 print(roc_auc_score(target2, forest.predict_proba(data2_red)[:,1]))


###Print importances of top features####
with open("glom_expr_filtered.csv") as textFile: 
 data_cor = [line.rstrip("\n").split(",") for line in textFile]

data_cor = [[float(y) for y in x] for x in data_cor]

with open("arff_files/feature_PAT_Cohort.tmp") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]

for i in range(0,len(target)):
 if target[i]=='MN':
  target[i] = 1
 else:
  target[i] = 0

forest = RandomForestClassifier(n_estimators=500, max_features=1000)
forest.fit(data_cor, target)
import numpy as np
indices = np.argsort(forest.feature_importances_)[::-1]

imp = forest.feature_importances_[indices[0:500]]
print '[%s]' % '\n'.join([str(i) for i in imp])
