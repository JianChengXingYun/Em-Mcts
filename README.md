# Em-Mcts

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2505.23229-b31b1b?style=flat-square )](https://arxiv.org/abs/2505.23229 )

<img width="1027" height="671" alt="企业微信截图_20260206090312" src="https://github.com/user-attachments/assets/eaff7000-bb40-491d-9d1b-e8a358a8eb0f" />

</div>

**Empirical-MCTS**: A dual-loop inference-time scaling framework that transforms stateless Monte Carlo Tree Search into continuous empirical learning. By unifying real-time meta-prompt evolution (PE-EMP) with global memory optimization, it enables LLMs to accumulate and reuse reasoning wisdom across problems—significantly boosting performance on complex reasoning benchmarks like AIME25 and MathArena Apex.

<div align="center">

<img width="484" height="839" alt="企业微信截图_20260206091122" src="https://github.com/user-attachments/assets/5232ff91-3b8f-482f-a4f0-2ee75824f549" />


<img width="1499" height="525" alt="企业微信截图_20260206091136" src="https://github.com/user-attachments/assets/451c5052-f1f3-4330-bfee-3a5c5728a9e9" />


</div>
---

## 🚀 Quick Start

### 1. Install Swimming_Pool Package

```bash
git clone https://github.com/JianChengXingYun/Em-Mcts 
cd Em-Mcts/Swimming_Pool_Async_project
python setup.py install
```

### 2. Launch Em-Mcts

```bash
cd Em-Mcts
python LLMExplorer_Socrates_em_mcts.py
```

---

## 📄 Paper

- **arXiv**: [https://arxiv.org/abs/2602.04248](https://arxiv.org/abs/2602.04248)

---

## 🤝 Contributing

We welcome community contributions! Feel free to open issues or submit pull requests.

---

## 📝 Citation

If you use this framework in your research, please cite our paper:

```bibtex
@misc{lu2026empiricalmctscontinuousagentevolution,
      title={Empirical-MCTS: Continuous Agent Evolution via Dual-Experience Monte Carlo Tree Search}, 
      author={Hao Lu and Haoyuan Huang and Yulin Zhou and Chen Li and Ningxin Zhu},
      year={2026},
      eprint={2602.04248},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.04248}, 
}
```
