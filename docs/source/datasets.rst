Example Datasets
=====

We utilized miRNA-seq datasets from three `TCGA studies <https://www.cancer.gov/tcga>`_:
Breast Invasive Carcinoma (BRCA), Prostate Adenocarcinoma (PRAD), Skin Cutaneous Melanoma (SKCM)

Within this package, there are two main folders of datasets. One set includes the datasets that can be used as a case study. These are the BRCA dataset with cancer subtypes (ILC, IDC) as group labels and marker filtering based on importance score from a random forest classification. 
Another dataset is the SKCMPositive_4 dataset which is the original SKCM that underwent marker filtering with 4 as the mean threshold on log scale.

The example datasets provided in the package can be used with ApplyExperiment for a few examples which can be found in :doc:`methods`. 
SKCMPositive_4 can be used for a pilot experiment using VAE with loss ratio 1-10.
A case study can be done on the dataset BRCASubtype using CVAE1-20.
Transfer learning as an example can be done using MAF from PRAD dataset to BRCA.