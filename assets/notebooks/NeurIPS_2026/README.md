# Graphons for Machine Learning: Theory, Estimation, and Applications

Jupyter worksheets for the NeurIPS 2026 Education Track mini-course *Graphons for Machine Learning: Theory, Estimation, and Applications*.

The worksheets let you run the examples covered in the course videos. Each worksheet matches one or more parts of the course.

## Contents

| Worksheet | Topic | Course part |
|---|---|---|
| `01_introduction_to_graphons.ipynb` | Sampling graphs from graphons; combining graphons (convex addition, disjoint union, disjoint union with a join) | Part 1 |
| `02_caron_fox_bnpgraph.ipynb` | Sparse graphs from exchangeable random measures (Kallenberg graphs) | Part 3 |
| `03_uw_mixtures.ipynb` | (U,W) graphon mixtures | Part 3 |
| `04_graphettes.ipynb` | Graphettes: sparse graphons with star and ring structures | Part 3 |
| `05_IGNR_single_example.ipynb` | Graphon estimation with IGNR (Implicit Graphon Neural Representation) | Part 4 |
| `06_jgs_graphon_estimation.ipynb` | Graphon estimation with JGS (Joint Graph Sorting) | Part 4 |

## Folder structure

```
NeurIPS_2026/
├── notebooks/            the six worksheets
├── src/                  helper modules used by the notebooks
│   ├── graphons.py
│   ├── Graphon_Mixtures_Graphettes.py
│   └── Caron_Fox_BNPGraph.py
├── requirements.txt
└── README.md
```

## Running in Google Colab

Open a worksheet in Colab and choose **Runtime, then Run all**. The first cell installs `python-igraph` and downloads the helper module it needs from this repository, so no other setup is needed.

Worksheets 5 and 6 also clone the original IGNR and JGS code from GitHub when they run. If they run slowly, choose **Runtime, then Change runtime type, then GPU**.

## Running locally

1. Install Python 3.9 or later.
2. Create a virtual environment and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install jupyter
   ```
3. Start Jupyter **from the `notebooks` folder**, so that the notebooks can find `../src`:
   ```bash
   cd notebooks
   jupyter notebook
   ```

Worksheets 5 and 6 need internet access when they first run, because they clone the IGNR and JGS repositories into the `notebooks` folder.

## Notes

- Worksheet 5 uses the original code from [Mishne-Lab/IGNR](https://github.com/Mishne-Lab/IGNR) and worksheet 6 uses [RolandBonifaceSogan/JGS-graphon](https://github.com/RolandBonifaceSogan/JGS-graphon). Please cite the original papers if you use them.
- The Caron and Fox module is a Python port of the MATLAB `BNPgraph` package by Francois Caron.

## References

- Caron, F. and Fox, E. B. (2017). Sparse graphs using exchangeable random measures. *Journal of the Royal Statistical Society: Series B*, 79(5), 1295-1366.
- Xia, X., Mishne, G. and Wang, Y. (2023). Implicit Graphon Neural Representation. *AISTATS 2023*, PMLR 206:10619-10634.
- Sogan, R. B. and Rebafka, T. (2026). Low-Complexity and Consistent Graphon Estimation from Multiple Networks. *AISTATS 2026*, PMLR 300:1441-1449.

## Acknowledgements

The Jupyter worksheets and the README were drafted with Claude (Anthropic). The author reviewed, tested and is responsible for all content.

## License

The code is released under the MIT license.
