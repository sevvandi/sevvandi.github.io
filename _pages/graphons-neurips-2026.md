---
layout: single
title: "Graphons for Machine Learning: Theory, Estimation, and Applications"
permalink: /teaching/graphons-neurips-2026/
author_profile: true
---

[← Back to Teaching]({{ '/teaching/' | relative_url }})

**Event:** *NeurIPS 2026 Education Track*
**Authors:** *Sevvandi Kandanaarachchi, Cheng Soon Ong*
**Date and location:** *Monday 7 December 2026, Sydney, Australia*

## Concept
Graphons are graph limits. Traditionally a graphon is defined on a unit square as $W:[0,1]^2 \to [0,1]$ with the output denoting the edge probability. They capture the asymptotic structure of a graph in a size-independent form, enabling graphs of different sizes to be studied and compared within a unified framework.

Graphons are increasingly used in machine learning: for GNN transferability across graphs of different sizes, and for neural network pruning, among other applications. Estimating a graphon from observed graph data is itself an active research problem, with newer approaches, such as IGNR and JGS,  offering more flexible, learnable alternatives to classical estimators. At the same time, while a rich body of theory has been developed for sparse graphons and related extensions, these models have seen relatively limited adoption within machine learning, despite the prevalence of sparse graphs in real-world applications.

Together, this combination of graphons' growing role in core ML tasks, a rapidly evolving landscape of estimation methods, and the largely untapped potential of sparse graphon models motivates their inclusion in the NeurIPS 2026 Education Track. This tutorial covers two state-of-the-art applications and emerging opportunities for leveraging graphons in machine learning.


**Prerequisites:** basic graph theory, linear algebra and calculus.

## Learning Objectives and Outcomes

On completion of this mini-course, participants will be able to:

1. understand the mathematical foundations of graphons, including homomorphism densities, cut distance/cut norm, convergence and exchangeability,

2. apply graphons as a base generative model to generate graphs, and combine graphons to construct richer and more complex network structures,

3. understand the limitations of classical graphons for modelling sparse graphs, and gain hands-on experience with sparse graph extensions covering Kallenberg exchangeable graphs, graphon mixtures, and graphettes,

4. evaluate and understand graphon estimation methods, including traditional methods such as SAS and more recent methods such as Implicit Graphon Neural Representation (IGNR) and the Joint Graph Sorting (JGS) estimator,

5. understand and describe how graphons are used in machine learning, for example in GNN transferability and neural network pruning.


## Recent papers using graphons in Machine Learning (at the time of writing)

1. Cheng, H., Chen, Y., Ma, P. and Zhong, W. (2026). Graphon Cross-Validation: Assessing Models on Network Data. In *The Fourteenth International Conference on Learning Representations*. [Link](https://iclr.cc/virtual/2026/poster/10011216)

2. Fan, X., Ma, F., Leng, C. and Wu, W. (2025). Low-Rank Graphon Learning for Networks. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*. [Link](https://proceedings.neurips.cc/paper_files/paper/2025/hash/17ea7ab0f55d6bd39b6c5afd30892576-Abstract-Conference.html)

3. Pham, H., Ta, T.-A., Jacobs, T., Burkholz, R. and Tran-Thanh, L. (2025). The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*.[Link](https://neurips.cc/virtual/2025/poster/119150)

4. Wang, Y., Cheng, Y.-H., Mukherjee, D. and Cheng, H. (2025). Transfer Learning on Edge Connecting Probability Estimation Under Graphon Model. In *The Thirty-ninth Annual Conference on Neural Information Processing Systems*.[Link](https://dl.acm.org/doi/10.52202/085713-1143)


## Teaching Materials 

Teaching materials consist of videos, slides and Jupyter worksheets detailed as follows:

### Part 1: Introduction to graphons
This video introduces graphons as a generative model and
samples graphs from different graphons. It combines graphons in 3 different ways: convex
addition, disjoint union and joining disjoint graphons.

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/S0bjvBGpm_M"
          title="Introduction to graphons"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen></iframe>
</div>
- [Slides (PDF)]({{ '/assets/slides/NeurIPS_2026/1_Graphons_Introduction.pdf' | relative_url }})
- Worksheet 1: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/01_introduction_to_graphons.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/01_introduction_to_graphons.ipynb' | relative_url }})



### Part 2: Mathematical properties of graphons
Mathematical properties of graphons: This video introduces graphons as graph limits
and explores convergence from the lens of homomorphism density and the cut metric.
Furthermore, it briefly discusses the assumption of exchangeability.

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/qBb9p_1lZNk"
          title="Mathematical properties of graphons"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen></iframe>
</div>
- [Slides (PDF)]({{ '/assets/slides/NeurIPS_2026/2_Mathematical_Properties_of_Graphons.pdf' | relative_url }})



### Part 3: Some approaches for sparse graphs
Sparse graph modelling with graphons: This video explores different sparse extensions
of traditional graphons.

<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/NkBkjGLPsfg"
          title="Sparse graph modelling with graphons"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen></iframe>
</div>
- [Slides (PDF)]({{ '/assets/slides/NeurIPS_2026/3_Sparse_Graph_Approaches.pdf' | relative_url }})

These 3 worksheets explore different graphon-related
sparse graph generation mechanisms including Kallenberg exchangeable graphs detailed in Caron and Fox (2017).

- Worksheet 2: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/02_caron_fox_bnpgraph.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/02_caron_fox_bnpgraph.ipynb' | relative_url }})
- Worksheet 3: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/03_uw_mixtures.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/03_uw_mixtures.ipynb' | relative_url }})
- Worksheet 4: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/04_graphettes.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/04_graphettes.ipynb' | relative_url }})



### Part 4: Graphon estimation
Graphon estimation: This video introduces graphon estimation when a single graph or
multiple graphs are observed.


<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/UGHZCvQ68H0"
          title="Graphon estimation"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen></iframe>
</div>
- [Slides (PDF)]({{ '/assets/slides/NeurIPS_2026/4_Graphon_Estimation.pdf' | relative_url }})

These 2 worksheets explores graphon estimation using IGNR (Xia et al, 2023) and JGS (Sogan and Rebafka, 2026).

- Worksheet 5: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/05_IGNR_single_example.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/05_IGNR_single_example.ipynb' | relative_url }})
- Worksheet 6: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/06_jgs_graphon_estimation.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/06_jgs_graphon_estimation.ipynb' | relative_url }})



### Part 5: Graphons in machine learning
Graphons in machine learning: This video discusses 2 applications of graphons – GNN
transferability and neural network pruning.
<div class="video-embed">
  <iframe src="https://www.youtube.com/embed/8yEbbxI-wy8"
          title="Graphons in machine learning"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen></iframe>
</div>
- [Slides (PDF)]({{ '/assets/slides/NeurIPS_2026/5_Graphons_in_ML.pdf' | relative_url }})




**Jupyter Worksheets**

1. Introduction to graphons: This worksheet introduces graphon sampling and combines
graphons in 3 different ways: convex addition, disjoint union and joining disjoint graphons. [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/01_introduction_to_graphons.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/01_introduction_to_graphons.ipynb' | relative_url }})

2. Kallenberg exchangeable graphs  (Caron and Fox, 2017):  [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/02_caron_fox_bnpgraph.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/02_caron_fox_bnpgraph.ipynb' | relative_url }})

3. Graphon mixtures:  [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/03_uw_mixtures.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/03_uw_mixtures.ipynb' | relative_url }})

4. Graphettes: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/04_graphettes.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/04_graphettes.ipynb' | relative_url }})

5. Graphon estimation with IGNR: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/05_IGNR_single_example.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/05_IGNR_single_example.ipynb' | relative_url }})

6. Graphon estimation with JGS: [Open in Colab](https://colab.research.google.com/github/sevvandi/sevvandi.github.io/blob/master/assets/notebooks/NeurIPS_2026/notebooks/06_jgs_graphon_estimation.ipynb) · [Download]({{ '/assets/notebooks/NeurIPS_2026/notebooks/06_jgs_graphon_estimation.ipynb' | relative_url }})


## Downloads
- [Slides (PDF)]({{ '/assets/slides/NeurIPS_2026/Neurips_Education_All.pdf' | relative_url }})
- [Mini-course write-up (PDF)]({{ '/assets/docs/Graphons_NeurIPS_2026_Writeup.pdf' | relative_url }})


[Download all worksheets (ZIP)]({{ '/assets/notebooks/NeurIPS_2026/worksheets.zip' | relative_url }})

## Acknowledgements

The Jupyter worksheets were developed with assistance from Anthropic's Claude. The first author reviewed, tested and is responsible for all content. The code in `Caron_Fox_BNPGraph.py` is a Python port of the MATLAB [`BNPgraph`](https://www.stats.ox.ac.uk/~caron/code/bnpgraph/demo_graph.html) package by Francois Caron, created with the help of Claude. It follows Caron and Fox (2017).

## References

1. Borgs, C., Chayes, J. T., Cohn, H. and Holden, N. (2018). Sparse exchangeable graphs and their limits via graphon processes. *Journal of Machine Learning Research*, 18(210), 1-71. [Link](https://www.jmlr.org/papers/v18/16-421.html)

2. Caron, F. and Fox, E. B. (2017). Sparse graphs using exchangeable random measures. *Journal of the Royal Statistical Society: Series B*, 79(5), 1295-1366. [Link](https://doi.org/10.1111/rssb.12233)

3. Kandanaarachchi, S. and Ong, C. S. (2026). Graphon mixtures. In *Proceedings of The 29th International Conference on Artificial Intelligence and Statistics*, PMLR 300:514-522. [Link](https://proceedings.mlr.press/v300/kandanaarachchi26a.html)

4. Lovász, L. (2012). *Large Networks and Graph Limits*. American Mathematical Society. [Link](https://www.sciencedirect.com/science/article/pii/S1574013713000294)

5. Pham, H., Ta, T.-A., Jacobs, T., Burkholz, R. and Tran-Thanh, L. (2025). The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis. In *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. [Link](https://neurips.cc/virtual/2025/poster/119150)

6. Ruiz, L., Chamon, L. and Ribeiro, A. (2020). Graphon Neural Networks and the Transferability of Graph Neural Networks. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. [Link](https://proceedings.neurips.cc/paper/2020/hash/12bcd658ef0a540cabc36cdf2b1046fd-Abstract.html)

7. Sogan, R. B. and Rebafka, T. (2026). Low-Complexity and Consistent Graphon Estimation from Multiple Networks. In *Proceedings of The 29th International Conference on Artificial Intelligence and Statistics*, PMLR 300:1441-1449. [Link](https://proceedings.mlr.press/v300/sogan26a.html)

8. Wijesinghe, A., Kandanaarachchi, S., Steinberg, D. M. and Ong, C. S. (2026). Flowette: Flow Matching with Graphette Priors for Graph Generation. In *The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026)*. [Link](https://openreview.net/forum?id=d3pAaxyrAY)

9. Xia, X., Mishne, G. and Wang, Y. (2023). Implicit Graphon Neural Representation. In *Proceedings of The 26th International Conference on Artificial Intelligence and Statistics*, PMLR 206:10619-10634. [Link](https://proceedings.mlr.press/v206/xia23a.html)
