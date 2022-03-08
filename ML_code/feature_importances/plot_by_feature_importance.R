####Plot AUC of top k features sorted by importance
library(ggplot2)
a <- read.table("AUCS_by_featurecount.txt", header=FALSE)
i = c(1:500)
ggplot() + geom_point(aes(i, a$V1)) + theme_Publication() + scale_y_continuous(limits=c(0.5,1)) + xlab("Number of Features") + ylab("AUC")  + theme(axis.text=element_text(size=18), axis.title=element_text(size=24,face="bold"))

dev.copy(svg, "feature_aucs_curve.svg")
dev.off()


####
#plot importances of top features
###

library(ggplot2)
a <- read.table("top_importances.txt", header=FALSE)
i = c(1:500)
ggplot() + geom_point(aes(i, a$V1)) + theme_Publication() +  xlab("Feature (Gene) Rank") + ylab("Importance")  + theme(axis.text=element_text(size=18), axis.title=element_text(size=24,face="bold"))

dev.copy(svg, "feature_importances.svg")
dev.off()
