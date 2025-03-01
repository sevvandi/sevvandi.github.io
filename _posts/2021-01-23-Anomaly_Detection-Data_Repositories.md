---
title: Anomaly detection data repositories
author: Sevvandi Kandanaarachchi
date: '2021-01-23'
output: 
  html_document:
    fig_path: "/assets/images/posts/"
    self_contained: true
    keep_md: true
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
This is a post I did in Jan 2021 on anomaly detection data repositories.

In this post we will look at data repositories available for anomaly detection. So, can you use a standard classification dataset for anomaly detection? You can if you *downsample* one class, preferably the minority class. You can label the downsampled observations as anomalies. If you're comparing performance for multiple anomaly detection methods, then one downsampled dataset is not enough. It is better to evaluate methods on multiple such versions to account for random sampling. If you're using anomaly detection methods that need numerical data, the categorical variables in your classification dataset has to be taken care of. Are you going to drop them or convert them to numerical data? In addition you need to take missing values into account.  So, let us look at some repositories. 

1. [Outlier Detection Datasets - ODDS](http://odds.cs.stonybrook.edu/) 
This is a really good website that provides multi-dimensional point datasets, time series graph datasets for event detection, time series point datasets, adversarial attack and security datasets and crowd scene video datasets. They describe the datasets well and provide references. It is maintained by Shebuti Rayana.

2. [Monash figshare outlier detection datasets](https://figshare.com/articles/dataset/Datasets_12338_zip/7705127) This repository has 12000 plus anomaly detection datasets that can be downloaded as a zip file. All these datasets are prepared using classification datasets from the UCI repostiory. This data repository was used in the paper *On normalization and algorithm selection for unsupervised outlier detection*.  

3. [ELKI outlier detection datasets](https://elki-project.github.io/datasets/outlier) This repository was used in the paper *On the Evaluation of Unsupervised Outlier Detection: Measures, Datasets, and an Empirical Study* by Campos et al. It contains over 2000 datasets prepared for anomaly detection. 

4. [Harvard dataverse anomaly detection datasets](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF) This respository contains 10 datasets prepared for anomaly detection. It was used in the paper *A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data* by Goldstein and Uchida. 

5. [ADRepository - Anomaly Detection Datasets with Real Anomalies](https://towardsdatascience.com/adrepository-anomaly-detection-datasets-with-real-anomalies-2ee218f76292)This is a GitHub repository maintained by Guansong Pang. It contains 21 datasets. More details are available in the paper *Deep learning for anomaly detection: a review* by Pang et al. 

6. [Anomaly Detection Meta-Analysis Benchmarks](https://ir.library.oregonstate.edu/concern/datasets/47429f155) This repository was used in the paper *A Meta-Analysis of the Anomaly Detection Problem* by Emmott et al. 

If I haven't listed your anomaly detection data repository, do let me know and I will include it.
  
