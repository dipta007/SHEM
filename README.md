# Semantically-informed Hierarchical Event Modeling

![Python Version](https://badgen.net/pypi/python/black)
![MIT License](https://img.shields.io/github/license/dipta007/SHEM?style=plastic)

This repository is the official implementation of [Semantically-informed Hierarchical Event Modeling](https://arxiv.org/abs/2212.10547) (Published in *SEM 2023).

![Main Figure](./figs/main.png)

## 👨🏻‍💻 Code

For simplicity and usability, the code is divided into different branches based on the experiments. The `main` branch contains the information of other branches. The following table shows the branches and their corresponding experiments.

| Branch | Experiment | Section in Paper | Table in Paper | Model Name in Paper |
| :---: | --- | :---: | :---: | :---: |
| [main](https://github.com/dipta007/SHEM) | Information on All experiments | - | - | - |
| [inference_frame](https://github.com/dipta007/SHEM/tree/inference_frame) | Is Frame Inheritance Sufficient? | 5.1 | 1, 6, 7 | `ours: inf. frame` |
| [lexical](https://github.com/dipta007/SHEM/tree/lexical) | Is Frame Inheritance Sufficient? | 5.1 | 1, 6, 7 | `ours: lexical` |
| [ind_frame](https://github.com/dipta007/SHEM/tree/ind_frame) | Relations Beyond Inheritance | 5.2 | 2, 8, 9 | `Using`, `Precedes`, `Metaphor`, `See_also`, `Causative_of`, `Inchoative_of`, `Perspective_on`, `Subframe`, `ReFraming_Mapping`  |
| [grp](https://github.com/dipta007/SHEM/tree/grp) | Relations Beyond Inheritance | 5.2 | 2, 12, 13 | `grouping` |
| [scn](https://github.com/dipta007/SHEM/tree/scn) | Relations Beyond Inheritance | 5.2 | 2, 10, 11 | `scenario_only` |
| [missing_grp](https://github.com/dipta007/SHEM/tree/missing_grp) | Predicting Missing Events | 5.3 | 3 | `grp` |
| [missing_scn](https://github.com/dipta007/SHEM/tree/missing_scn) | Predicting Missing Events | 5.3 | 3 | `scn` |
| [event_similarity](https://github.com/dipta007/SHEM/tree/event_similarity) | Improved Event Similarity | 5.4 | 4 | `ours` |


## ✉️ Contact

If you want to contact me you can reach me at [@Shubhashis Roy Dipta](mailto:sroydip1@umbc.edu) or open an issue in this repository.


## ⚠️ Disclaimer

Some parts of the code were inspired by [SSDVAE](https://github.com/mmrezaee/SSDVAE) implementations.
