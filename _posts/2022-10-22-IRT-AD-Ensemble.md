---
title: Anomaly Detection Ensembles
author: Sevvandi Kandanaarachchi
date: '2022-10-22'
output: 
  html_document:
    fig_path: "/assets/images/posts/"
    self_contained: true
    keep_md: true
layout: single
image:
  caption: ''
  focal_point: ''
  preview_only: yes
toc: true
mathjax: true     
---

This is a post I did in Oct 2022 on using Item Response Theory to construct and anomaly detection ensemble.

## What is an anomaly detection ensemble?

It is a bunch of anomaly detection methods put together to get a final anomaly score/prediction. So you have a bunch of methods, and each of these methods have its own anomaly score, which is used by the ensemble to come up with the consensus score. 


What are the ways of constructing an anomaly detection ensemble? Broadly, anomaly detection ensembles can be categorised into 3 camps.

1. Feature bagging
2. Subsampling
3. Using combination functions


## Feature bagging
Feature bagging is a very popular ensemble technique in anomaly detection. Feature bagging uses different attribute subsets to find anomalies. In a dataset, generally observations are denoted by rows and attributes are denoted by columns. Feature bagging considers different column subsets. That is, multiple copies of the same dataset each having a slightly different set of columns is considered. For each dataset copy, we find anomalies using a single anomaly detection method.  Then the anomaly scores are averaged to compute the ensemble score. 

Let us try this with the letter dataset from the [ODDS repository](http://odds.cs.stonybrook.edu/). We first read the dataset and normalize it so that each column has values within 0 and 1. Let's have a loot at the data after normalising.  




```r
datori <- readMat("letter.mat")
Xori <- datori$X
Xori <- unitize(Xori)
head(Xori)
```

```
##            [,1]      [,2]       [,3] [,4]       [,5] [,6]      [,7]       [,8]
## [1,] 0.40000000 0.6666667 0.33333333  0.6 0.20000000  0.7 0.3076923 0.30769231
## [2,] 0.00000000 0.4000000 0.00000000  0.4 0.00000000  0.4 0.3846154 0.30769231
## [3,] 0.26666667 0.4666667 0.33333333  0.5 0.20000000  0.4 0.4615385 0.15384615
## [4,] 0.06666667 0.4000000 0.06666667  0.4 0.13333333  0.4 0.3846154 0.00000000
## [5,] 0.06666667 0.1333333 0.06666667  0.3 0.06666667  0.4 0.3846154 0.07692308
## [6,] 0.06666667 0.3333333 0.00000000  0.7 0.00000000  0.4 0.3846154 0.30769231
##      [,9] [,10]     [,11]     [,12]     [,13]     [,14]     [,15]     [,16]
## [1,]  0.6   1.0 0.1818182 0.3636364 0.1333333 0.6428571 0.3636364 0.8888889
## [2,]  0.4   0.3 0.4545455 0.5454545 0.0000000 0.5714286 0.0000000 0.5555556
## [3,]  0.7   0.3 0.4545455 0.6363636 0.0000000 0.5714286 0.3636364 0.5555556
## [4,]  0.7   0.3 0.4545455 0.5454545 0.0000000 0.5714286 0.2727273 0.5555556
## [5,]  0.7   0.3 0.4545455 0.5454545 0.0000000 0.5714286 0.2727273 0.5555556
## [6,]  0.4   0.3 0.4545455 0.5454545 0.0000000 0.5714286 0.0000000 0.5555556
##           [,17]     [,18]      [,19]     [,20]      [,21] [,22]     [,23]
## [1,] 0.26666667 0.6666667 0.33333333 0.8888889 0.14285714   0.5 0.5714286
## [2,] 0.20000000 0.4666667 0.26666667 0.5555556 0.14285714   0.5 0.4285714
## [3,] 0.06666667 0.2666667 0.00000000 0.2222222 0.00000000   0.5 0.4285714
## [4,] 0.13333333 0.1333333 0.06666667 0.3333333 0.07142857   0.5 0.4285714
## [5,] 0.06666667 0.2666667 0.06666667 0.3333333 0.07142857   0.5 0.4285714
## [6,] 0.06666667 0.6666667 0.00000000 0.7777778 0.00000000   0.5 0.4285714
##           [,24]     [,25]     [,26] [,27]     [,28] [,29]     [,30]      [,31]
## [1,] 0.00000000 0.5714286 1.0000000   0.5 0.3076923     0 0.7500000 0.18181818
## [2,] 0.00000000 0.5000000 0.9285714   0.5 0.4615385     0 0.5833333 0.09090909
## [3,] 0.07142857 0.5000000 0.5000000   0.5 0.4615385     0 0.5833333 0.18181818
## [4,] 0.07142857 0.5714286 0.5000000   0.5 0.4615385     0 0.5833333 0.27272727
## [5,] 0.07142857 0.5714286 0.5000000   0.5 0.5384615     0 0.5833333 0.27272727
## [6,] 0.28571429 0.2857143 0.5000000   0.5 0.4615385     0 0.5833333 0.00000000
##      [,32]
## [1,]   0.5
## [2,]   0.5
## [3,]   0.6
## [4,]   0.6
## [5,]   0.6
## [6,]   0.6
```

Now, feature bagging would select different column subsets. Let's pick different columns.




```r
set.seed(1)
dd <- dim(Xori)[2]
sample_list <- list()
for(i in 1:10){
  sample_list[[i]] <- sample(1:dd, 20)
}
sample_list[[1]]
```

```
##  [1] 25  4  7  1  2 23 11 14 18 19 29 21 10 32 20 30  9 15  5 27
```

```r
sample_list[[2]]
```

```
##  [1]  9 25 14  5 29  2 10 31 12 15  1 20  3  6 26 18 19 23  4 24
```

Next we select the subset of columns in each sample_list and find anomalies in each subsetted-dataset. Let's use the KNN_AGG anomaly detection method. This method aggregates the k-nearest neighbour distances. If a data point has high KNN distances compared to other points, then it is considered anomalous, because it is far away from other points. 


```r
library(DDoutlier)
knn_scores <- matrix(0, nrow = NROW(Xori), ncol = 10)
for(i in 1:10){
  knn_scores[ ,i] <- KNN_AGG(Xori[ ,sample_list[[i]]])
}
head(knn_scores)
```

```
##           [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]
## [1,] 21.635185 22.270694 24.987923 20.026546 22.158682 22.422737 20.455542
## [2,]  9.969831  9.661039  9.432406  5.255449  5.540366  7.325621  9.302664
## [3,] 16.061851 20.796525 11.832477 14.235700 11.481942 14.899982 13.096920
## [4,] 12.409446 14.012729 11.400507  7.347249  7.739870  9.309485  9.793680
## [5,]  9.998477 12.541672 10.209202  6.857942  5.919889  9.355743  8.214986
## [6,] 12.350209 12.350209  7.409608  4.905166  5.045559  3.310723  7.461390
##          [,8]      [,9]     [,10]
## [1,] 23.55476 21.493597 25.954645
## [2,] 10.51396  7.460010  8.571801
## [3,] 17.98096  9.872211 19.625926
## [4,] 11.78520  7.757524 11.292598
## [5,] 12.24067  6.955316 11.085984
## [6,] 12.29468  3.638703  7.127627
```

Now we have the anomaly scores for the 10 subsetted-datasets. In feature bagging the general method of consensus is to add up the scores or take the mean of the scores, which is an equivalent thing to do. 


```r
bagged_score <- apply(knn_scores, 1, mean)
```

We can compare the bagged anomaly scores with the anomaly scores f we didn't use bagging. That is, if we used the full dataset, what would be anomaly scores? Does bagging make it better?  For this we need the labels/ground truth. To evaluate the performance, we use the area under the ROC curve.


```r
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
```

```
## 
## Attaching package: 'pROC'
```

```
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

```r
labels <- datori$y[ ,1]

# anomaly scores without feature bagging - taking the full dataset
knn_agg_without <- KNN_AGG(Xori)

# ROC  - without bagging
rocobj1 <- roc(labels, knn_agg_without, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.9097
```

```r
rocobj2 <- roc(labels, bagged_score, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj2$auc
```

```
## Area under the curve: 0.9101
```
Yes! We see that there is an increase in AUC (area under the ROC curve) by feature bagging. In this case it is a small improvement. But, nonetheless there is an improvement.


## Subsampling
Subsampling uses different subsets of observations to come up with anomaly scores. Instead of columns, here we use different subsets of observations. Then we average the anomaly scores to get an ensemble score. First, let's get the different observation samples.  For this, we will use non-anomalous observations because the anomalous observations are rare, we don't want to use all of them. 


```r
set.seed(1)
sample_matrix <- matrix(0, nrow = NROW(Xori), ncol = 10)
inds0 <- which(labels == 0)
nn1 <- sum(inds0)
inds1 <- which(labels == 1)
nn2 <- sum(inds1)
sample_matrix[inds1, ] <- 1
for(j in 1:10){
  sam <- sample(inds0, 1400)
  sample_matrix[sam, j] <- 1
}
head(sample_matrix)
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
## [1,]    1    1    1    1    0    1    1    1    1     1
## [2,]    1    1    1    1    1    1    1    1    1     1
## [3,]    1    1    1    1    1    1    1    1    1     1
## [4,]    1    1    1    1    1    1    1    1    1     1
## [5,]    1    1    1    0    1    1    1    1    1     1
## [6,]    0    1    1    1    1    1    0    1    1     1
```

Our sample_matrix contains 1 if that observation is going to be used and 0 if it doesn't. We are going to use 10 subsampling iterations.

Now that we have our subsamples, let's use an anomaly detection method to get the anomaly scores.


```r
anom_scores <- matrix(NA, nrow = NROW(Xori), ncol = 10)
for(j in 1:10){
  inds <- which(sample_matrix[ ,j] == 1)
  Xsub <- Xori[inds, ]
  anom_scores[inds,j] <- KNN_AGG(Xsub)
}
head(anom_scores)
```

```
##          [,1]     [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
## [1,] 30.13047 31.70672 31.38430 30.13047       NA 30.13047 30.13047 30.13047
## [2,] 11.46522 11.46522 11.51939 11.70227 11.63900 12.39053 11.46522 11.46522
## [3,] 22.95041 23.53152 22.95041 23.07765 22.95041 22.95041 23.67323 23.52891
## [4,] 15.17929 16.05668 15.17929 15.76817 15.17929 15.78328 16.57122 15.44892
## [5,] 13.57168 15.51147 13.60095       NA 13.57168 15.51147 14.02113 15.04368
## [6,]       NA 13.46472 12.58007 13.07503 13.07503 12.70142       NA 12.58007
##          [,9]    [,10]
## [1,] 30.13047 30.13047
## [2,] 11.46522 11.51939
## [3,] 23.07765 22.95041
## [4,] 15.20815 15.20815
## [5,] 15.04368 15.04368
## [6,] 12.75612 12.66090
```

We see there are NA values when that observation was not selected. Now we will get the mean anomaly score. But some observations did not got selected for certain iterations. We need to take that into account. 


```r
rowsum <- apply(sample_matrix, 1, sum)
subsampled_score <- apply(anom_scores, 1, function(x) sum(x, na.rm = TRUE))/rowsum
head(subsampled_score)
```

```
## [1] 30.44492 11.60967 23.16410 15.55824 14.54660 12.86167
```

Here, we have divided the sum of the anom_scores by the number of times each observation was selected. The ensemble score is subsampled_score.  Now we can see if the ensemble score is better than the original score. 


```r
# ROC  - without bagging
rocobj1 <- roc(labels, knn_agg_without, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.9097
```

```r
rocobj2 <- roc(labels, subsampled_score, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj2$auc
```

```
## Area under the curve: 0.9073
```
Oh dear! Not really for this example. But it doesn't go down by much, which is a relief. Sometimes the ensemble is not better than the original model.  But most of the time it is. That is why we use ensembles. 

## Using combination functions
For both the above examples we used the average as the combination function. That is, given a set of scores for each observation, we averaged them. But we can do different things. For example, we can get the maximum instead of the average. Or we can get the geometric mean which is multiplying all of the scores and getting the Nth root of them. As examples, let's try those two functions: the maximum and the geometric mean. Let's use knn_scores in the bagging example. 


```r
max_score <- apply(knn_scores, 1, max)
head(max_score)
```

```
## [1] 25.95465 10.51396 20.79653 14.01273 12.54167 12.35021
```

```r
prod_score <-(apply(knn_scores, 1, prod))^(1/10)
head(prod_score)
```

```
## [1] 22.427853  8.097505 14.597109 10.059545  9.070620  6.828194
```

Let's see if this is better than taking the average (mean). 


```r
rocobj1 <- roc(labels, bagged_score, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.9101
```

```r
# ROC  - Max
rocobj1 <- roc(labels, max_score, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.9129
```

```r
rocobj2 <- roc(labels, prod_score, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj2$auc
```

```
## Area under the curve: 0.908
```
Well!  Taking the maximum improved it a bit, but taking the geometric mean reduced it a bit. Interesting! 


But, combination functions are not generally used this way! In this example we used feature bagging anomaly scores and different combination functions. And we only used a single anomaly detection method all the time. Generally, combination functions are used when multiple anomaly detection methods are used. That way, it makes sense to weight  methods differently to get an ensemble score. 
## Combination functions with multiple anomaly detection methods
So far, we've only looked at one anomaly detection method. Let's take multiple anomaly detection methods from the DDoutlier package. 


```r
knn <- KNN_AGG(Xori)
lof <- LOF(Xori)
cof <- COF(Xori)
inflo <- INFLO(Xori)
kdeos <- KDEOS(Xori)
ldfsscore <- LDF(Xori)
ldf <- ldfsscore$LDE
ldof <- LDOF(Xori)
```

Now we have tried 7 anomaly detection methods. Let's see different combination functions. 

1. Mean
2. Maximum
3. Geometric Mean
4. An IRT-based method

The IRT-based method is discussed in detail in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025521012639). Basically, there is a fancier combination method which uses item response theory under the hood. 

Let's try these methods and get ensemble scores. 



```r
library(outlierensembles)

anomaly_scores <- cbind.data.frame(knn, lof, cof, inflo, kdeos, ldf, ldof)
mean_ensemble <- apply(anomaly_scores, 1, mean)
max_ensemble <- apply(anomaly_scores, 1, max)
geom_mean_ensemble <- (apply(anomaly_scores, 1, prod))^(1/7)
irt_mod <- irt_ensemble(anomaly_scores)
```

```
## Warning in sqrt(diag(solve(Hess))): NaNs produced

## Warning in sqrt(diag(solve(Hess))): NaNs produced

## Warning in sqrt(diag(solve(Hess))): NaNs produced

## Warning in sqrt(diag(solve(Hess))): NaNs produced

## Warning in sqrt(diag(solve(Hess))): NaNs produced
```

```r
irt_ensemble <- irt_mod$scores
```

Let's evaluate the different ensembles. 


```r
rocobj1 <- roc(labels, mean_ensemble, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.76
```

```r
# ROC  - Max
rocobj1 <- roc(labels, max_ensemble, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.7613
```

```r
rocobj1 <- roc(labels, geom_mean_ensemble, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.406
```

```r
rocobj1 <- roc(labels, irt_ensemble, direction = "<")
```

```
## Setting levels: control = 0, case = 1
```

```r
rocobj1$auc
```

```
## Area under the curve: 0.09039
```

For this example, IRT ensemble performs best. The mean and the max ensembles perform similarly. The geometric mean ensemble performs very poorly. I just included the geometric mean as it is another function, but actually it is not used in anomaly detection ensembles. 

As you see, there are different methods of ensembling. You can use feature bagging, subsampling or you can use a different combination function. And you can do more than one thing. You can do feature bagging and  use a different combination function. You can do bagging and subsampling together. A lot of options! 
