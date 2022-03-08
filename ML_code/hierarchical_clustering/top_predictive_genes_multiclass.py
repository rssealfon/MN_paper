##Multiclass classifier to identify top 500 genes predictive of diagnosis
import numpy as np
from sklearn.ensemble import RandomForestClassifier

with open("glom_expr_filtered.csv") as textFile: 
 data_cor = [line.rstrip("\n").split(",") for line in textFile]

data_cor = [[float(y) for y in x] for x in data_cor]

with open("feature_PAT_Cohort.tmp") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]

forest = RandomForestClassifier(n_estimators=500, max_features=1000)

all_importances= np.empty((0, 23579))

for i in range(0,10):
 forest.fit(data_cor,target)
 importances = forest.feature_importances_
 all_importances = np.append(all_importances,[importances], axis=0)
 print(i)

m = np.median(all_importances, axis=0)

with open("glom_genes.txt") as f:
 glom_genes = [line.rstrip('\n').rstrip('\r') for line in f]

indices = np.argsort(m)[::-1]
n = np.take(glom_genes, indices[0:500])

f = open("neptune_top_rf_cohort_features_med.txt", "w")

for gene in n:
 f.write("%s\n" % gene)

f.close()
