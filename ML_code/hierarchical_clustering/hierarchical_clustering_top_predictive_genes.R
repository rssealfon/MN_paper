##Hierarchical clustering plot of 500 genes most predictive of diagnosis
library(swamp)
genes <- read.table("neptune_top_rf_cohort_features_med.txt", header=FALSE)$V1 #top 500 genes in cohort classifier
a <- read.table("glom_expr_filtered.csv", sep = ",", header=FALSE)
genes <- as.character(genes)

all_genes <- read.table("glom_genes.txt")$V1
all_genes <- as.character(all_genes)

a_filt <- t(a)[all_genes %in% genes,]

dis=read.table("feature_PAT_Cohort.tmp")
rownames(dis) <- colnames(a_filt)
names(dis) <- c("diagnosis")

colnames(a_filt) <- rownames(dis)

dis$diagnosis <- revalue(dis$diagnosis, c("FSGS" = "Other", "IgA" = "Other", "MCD" = "Other"))

palette(c("#FEE9C8","#E34A33"))

hca.plot(a_filt,dis)
dev.copy(pdf, "hca_plot_nept_med.pdf")
dev.off()