if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

#BiocManager::install("curatedTCGAData")
#BiocManager::install("TCGAutils")
#BiocManager::install("MultiAssayExperiment")

library(curatedTCGAData)
library(TCGAutils)
library(MultiAssayExperiment)

#–– 1) Fetch the SARC dataset with normalized RNA-seq ––
# Specify the “SARC” disease code, the RNASeq2GeneNorm assay, and a valid version  [oai_citation:0‡Bioconductor](https://www.bioconductor.org/packages/release/data/experiment/vignettes/curatedTCGAData/inst/doc/curatedTCGAData.R)
mae <- curatedTCGAData(
  diseaseCode = "SARC",
  assays      = "RNASeq2GeneNorm",
  version     = "2.1.1",
  dry.run     = FALSE
)

#–– 2) Extract the SummarizedExperiment with clinical cols appended ––
# Find the exact assay name (it will include “RNASeq2GeneNorm”) and then pull it with clinical data  [oai_citation:1‡Bioconductor](https://www.bioconductor.org/packages/release/data/experiment/vignettes/curatedTCGAData/inst/doc/curatedTCGAData.R)
assay_name <- grep("RNASeq2GeneNorm", names(experiments(mae)), value=TRUE)
se         <- getWithColData(mae, assay_name, mode="append")

#–– 3) Build X: gene-expression matrix (genes × ASPS samples) ––
X <- assay(se)

#–– 4) Build surv from the same colData ––
# Use TCGAutils to get the standard clinical field names for SARC  [oai_citation:2‡Bioconductor](https://www.bioconductor.org/packages/devel/bioc/manuals/TCGAutils/man/TCGAutils.pdf?utm_source=chatgpt.com)
clin_vars <- getClinicalNames("SARC")
cd        <- as.data.frame(colData(se)[, clin_vars])

# Create follow-up time (days): days_to_death if event else days_to_last_followup
cd$time   <- ifelse(is.na(cd$days_to_death), cd$days_to_last_followup, cd$days_to_death)
cd$status <- cd$vital_status  # integer: 1 = dead, 0 = censored

# subset X and cd by their overlap
common_samples <- intersect(colnames(X), rownames(cd))
X <- X[, common_samples]
cd <- cd[common_samples, ]
# for X, keep 1000 rows with the highest variance, do not consider NA
X <- X[order(apply(X, 1, var), decreasing = TRUE)[1:1000], ]
# check how many NAs in each row of X
na_counts <- rowSums(is.na(X))
sum(na_counts)

#quick look at the dataset
mydata=cbind(cd, t(X))
head(mydata)
