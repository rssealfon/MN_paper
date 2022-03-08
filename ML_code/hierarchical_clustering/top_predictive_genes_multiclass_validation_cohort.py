##Multiclass classifier to identify top 500 genes predictive of diagnosis in validation cohort
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

with open("validation_expr.txt") as textFile: 
    data_cor_expanded = [line.rstrip("\n").split("\t") for line in textFile]
data_cor_expanded = data_cor_expanded[1:len(data_cor_expanded)]
with open("validation_cohort.txt") as f: 
 target = [line.rstrip('\n').rstrip('\r') for line in f]
data_cor_expanded = np.transpose(data_cor_expanded)
forest = RandomForestClassifier(n_estimators=500, max_features=1000)
all_importances= np.empty((0, 37293))
for i in range(0,10):
 forest.fit(data_cor_expanded,target)
 importances = forest.feature_importances_
 all_importances = np.append(all_importances,[importances], axis=0)
m = np.median(all_importances, axis=0)
indices = np.argsort(m)[::-1]
data_cor_expanded2 = data_cor_expanded.astype(np.float)
data_cor_expanded2[:,indices[0:500]].tofile("top_500_genes_multiclass.txt", sep=",")