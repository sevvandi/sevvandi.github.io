---
title: Anomaly detection dilemmas
author: Sevvandi Kandanaarachchi
date: '2020-07-19'
output: 
  html_document:
    fig_path: "/assets/images/posts/"
    self_contained: true
    keep_md: true
image:
  caption: ''
  focal_point: ''
  preview_only: yes
---

This is a blog post I did in July 2020 on challenges in anomaly detection.

Finding anomalies/outliers in data is a task that is increasingly getting more attention mainly due to the variety of applications involved. Not that it is a new field of research. Rather, outlier/anomaly detection has been studied by statisticians and computer scientists for a long time. However, there are certain aspects which lack consensus. 

1. **What's in a name?** 

The words outlier/anomaly/novelty/extremes are sometimes used to describe the same thing; sometimes for slightly different things. So, when we read a paper, we need to be careful what the word means, because it may not mean what we think it means.  

2. **Lack of definitions**

Antony Unwin in his paper *Multivariate outliers and the O3 plot* says *"Outliers are a complicated business. It is difficult to define what they are, it is difficult to identify them, and it is difficult to assess how they affect analyses".* Sometimes we find that the definition of an outlier depends on the application. For example, chromosomal anomalies in tumours may have very different distributional properties when compared with fraudulent credit card transactions among billions of legitimate transactions. This makes it difficult if not impossible for researchers to come up with algorithms that detect anomalies in all situations. Often, parameter selection plays a role. For example, if we are using an anomaly detection method based on knn distances, frequently we're faced with the question "which k works best?". 


3. **Identify outliers or give scores?**

Researchers broadly detect outliers in two different ways. (1) Identify outliers, i.e. declare if a data point is an outlier. This is a binary declaration. (2) Give a score of outlyingness. With this method, each point in the dataset gets a score of outlyingness. Both ways have pros and cons. The binary identification of outliers does not tell you how outlying it is. With this technique we have no sense which point is the most outlying of the identified outliers. On the other hand, the scoring method does not tell you which points are actually outliers. That is left to the user. The user can define a threshold and say that points with outlying scores above that threshold are outliers. However, coming up with a meaningful threshold may not be an easy task. 

4. **Evaluation methods**

Again, there is no consensus on how to evaluate anomaly detection methods. This is not surprising given that outlier detection methods have two modes of operation: identify outliers or give scores. Methods that identify outliers evaluate the anomaly detection method by using metrics such as true positive rate, false positive rate, positive predictive power and negative predictive power. On the other hand, anomaly detection methods that give scores use area under the Receiver Operator Characteristic (ROC) curve, or area under the Precision Recall curve to evaluate effectiveness.  

5.  **Datasets, or the lack there of**. 

I believe it is fair to say that until recently there were no benchmark datasets for anomaly detection or at least little was known about such datasets. In the recent past there has been some attempt to fill this gap and now there are some repositories of datasets specially prepared for anomaly detection. Even now, most of the datasets that are available are downsampled classification datasets. Yes, we know the ground truth of these datasets, but samples of the 'other class' may not be real anomalies.


