# HLCL: Graph Contrastive Learning under Heterophily via Graph Filters

Official minimal implementation of **HLCL** (UAI 2024). HLCL introduces a homophily–heterophily separation and applies **low-pass** and **high-pass graph filters** to augmented views, enabling state-of-the-art graph contrastive learning on heterophilic graphs.

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
