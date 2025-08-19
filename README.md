# HLCL: Graph Contrastive Learning under Heterophily via Graph Filters

Official minimal implementation of **HLCL** (UAI 2024). HLCL introduces a homophily–heterophily separation and applies **low-pass** and **high-pass graph filters** to augmented views, enabling state-of-the-art graph contrastive learning on heterophilic graphs.

## Paper
Wenhan Yang, Baharan Mirzasoleiman.  
[Graph Contrastive Learning under Heterophily via Graph Filters](https://arxiv.org/abs/2303.06344) (UAI 2024).

> HLCL identifies homophilic and heterophilic subgraphs using feature similarity, applies low-pass filters to the homophilic subgraph and high-pass filters to the heterophilic subgraph, and learns node representations by contrasting filtered augmented views. This approach achieves up to a **7% boost** over prior CL methods on heterophilic graphs and up to **10% over supervised baselines**.


## Installation

```bash
pip install -r requirements.txt
````

## Usage

```bash
python HLCL.py \
  --device 2 \
  --dataset pubmed \
  --low_k 0.8 \
  --high_k 0.2 \
  --augmentation PPRDiffusion \
  --haug1 0.4 \
  --runs 3 \
  --num_parts 3
```

Results and checkpoints are saved in:

* `./result/`
* `./model/`

## Citation

```bibtex
@inproceedings{yang2024hlcl,
  title={Graph Contrastive Learning under Heterophily via Graph Filters},
  author={Yang, Wenhan and Mirzasoleiman, Baharan},
  booktitle={Uncertainty in Artificial Intelligence (UAI)},
  year={2024}
}
```

```
```
