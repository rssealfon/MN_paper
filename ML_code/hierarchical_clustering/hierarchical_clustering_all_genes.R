####Hierarchical clustering plot with all genes######
###Note: expression data available in GEO, cohort labels available in Table S1

library(data.table)
library(swamp)
a <- fread("glom_expr_filtered.csv", sep = ",", header=FALSE)
dis=read.table("feature_PAT_Cohort.tmp")
a <- t(a)
colnames(a) <- rownames(dis)
names(dis) <- c("diagnosis")
all_genes <- read.table("glom_genes.txt")$V1
library(plyr)
dis$diagnosis <- revalue(dis$diagnosis, c("FSGS" = "Other", "IgA" = "Other", "MCD" = "Other"))
palette(c("#FEE9C8","#E34A33"))
hca.plot(a,dis)
dev.copy(pdf, "all_genes_diagnosis_heatmap.pdf")
dev.off()
