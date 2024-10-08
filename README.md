# RAE: Retrieval-enhanced Knowledge Editing for Multi-Hop Question Answering

This repository contains the official implementation of our CIKM'2024 paper "[Retrieval-enhanced Knowledge Editing in Language Models for Multi-Hop Question Answering](https://arxiv.org/abs/2403.19631)" by Yucheng Shi, Qiaoyu Tan, Xuansheng Wu, Shaochen Zhong, Kaixiong Zhou, Ninghao Liu.

## Overview

RAE is a novel framework for editing knowledge in large language models (LLMs) for multi-hop question answering tasks. It employs mutual information maximization for fact retrieval and a self-optimizing technique to prune redundant data.
![RAE](https://github.com/sycny/sycny.github.io/blob/main/images/work_flowv5.png)

## Data

### MQUAKE-CF-3k and MQUAKE-T Edited Knowledge Graph (KG)

You can download from these links: [KG_MQUAKE-CF-3k](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/Ec0O9oUzka5LuNwK3M8FL-YBG3zw7mAdme7V9S9l4cbt7Q?e=3TKvcE) and [KG_MQUAKE-T](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EckS-8zKM75MgqmJmQH8NQMByT___C5lNyZaIsOXHQXvIQ?e=VVwF2F).

Put them into `./data/`.

### Editing from Other Datasets

To build your edited KG :

```bash
python edit_KG.py
```

Note: You need to first download the original Wikidata KG from [here](https://outlookuga-my.sharepoint.com/:u:/g/personal/ys07245_uga_edu/EbbXuq1FumtFkH3B0qmb2bMBgRdyXbayUNAevKsKvtBVUw?e=Xy3QsY). This Wikidata KG is based on the [Wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m) project.

## Dependencies

Please refer to `requirements.txt` for the list of dependencies.

## Running the Code

Ensure you have prepared the edited KG before running:

### Editing on MQUAKE-CF-3k

```bash
python main.py --model gpt2 --mode beam --dataset MQuAKE-CF-3k
```

### Editing on MQUAKE-T

```bash
python main.py --model gpt2 --mode beam --dataset MQuAKE-T
```

## Arguments Explanation

- `NatureL`: When enabled, transforms a triple into a human-readable natural language statement. This benefits LLM modeling and improves retrieval success. Enabled by default.

- `Template`: When enabled, builds "question+fact chain" as in-context examples to help LLMs understand the task. Examples are extracted from MQUAKE-CF, containing 9k examples different from the test cases.

- `Template_number`: Number of templates used to extract relevant facts for fact chain retrieval. Default is 3.

- `entropy_template_number`: Number of templates used for knowledge pruning tasks. Default is 6.

- `correctConflict`: Specific design for MQUAKE-CF-3k dataset to handle editing conflicts where both unedited and edited versions of a fact are needed to answer different questions. You can leanr more details about this issue from [DeepEdit](https://arxiv.org/abs/2401.10471). Enabled by default but not necessary for other datasets. 


## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{shi2024retrieval,
  title={Retrieval-enhanced knowledge editing for multi-hop question answering in language models},
  author={Shi, Yucheng and Tan, Qiaoyu and Wu, Xuansheng and Zhong, Shaochen and Zhou, Kaixiong and Liu, Ninghao},
  journal={arXiv preprint arXiv:2403.19631},
  year={2024}
}
```
