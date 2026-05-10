# LoRA: Low-Rank Adaptation on RoBERTa-base / SST-2

Re-implementation of **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) for CS 4782 (Cornell, Spring 2026).

**Authors:** Elliott Lin, Anne Xia, Trisha Saini, Samarth Shridhar.

## 1. Introduction

LoRA freezes a pretrained Transformer's weights $W_0$ and learns a low-rank additive update $\Delta W = BA$ with rank $r \ll d$, so the forward pass becomes $h = W_0 x + (\alpha/r)\, BAx$. Only $A$ and $B$ are trained; $W_0$ stays frozen. The paper's main claim is that this matches or beats full fine-tuning on GLUE with orders of magnitude fewer trainable parameters and no added inference latency (since $BA$ can be folded back into $W_0$ at deployment).

This repo reproduces that claim on **RoBERTa-base / SST-2** and adds a joint rank/alpha ablation.

## 2. Chosen Result

We reproduce the RoBERTa-base row of **Table 2** from the paper on the SST-2 sentiment classification task:

- Full fine-tuning: **94.8%** val accuracy, 125M trainable params (paper)
- LoRA $r = 8$: **95.1%** val accuracy, 0.3M trainable params (paper)

We extend the paper's experiments with a joint sweep over rank $r \in \{2, 4, 6, 8\}$ and scaling factor $\alpha \in \{2, 4, 6, 8\}$ to investigate how the $\alpha/r$ ratio affects training stability.

## 3. GitHub Contents

```
.
├── code/
│   └── LoRA_RoBERTa_Classification.ipynb   # main notebook (Colab)
├── data/
│   └── README.md                            # SST-2 auto-downloads from HuggingFace
├── results/
│   ├── csv_output/                          # per-(rank, alpha) training/testing CSVs
│   │   ├── rank = 2, alpha = 2/
│   │   ├── rank = 2, alpha = 4/
│   │   ├── rank = 2, alpha = 6/
│   │   ├── rank = 2, alpha = 8/
│   │   ├── rank = 4, alpha = 4/
│   │   ├── rank = 4, alpha = 8/
│   │   ├── rank = 6, alpha = 6/
│   │   ├── rank = 6, alpha = 8/
│   │   └── rank = 8, alpha = 8/
│   └── figures/                             # accuracy / loss plots, ablation charts
├── poster/
│   └── LoRA_Poster.pdf
├── report/
├── LICENSE                                  # MIT
├── .gitignore
└── README.md
```

## 4. Re-implementation Details

- **Model.** RoBERTa-base (125M params) from HuggingFace `FacebookAI/roberta-base`.
- **LoRA adapter.** A `LoRA_Adapter` class wraps each `nn.Linear` and adds trainable matrices $A \in \mathbb{R}^{r \times d_\text{in}}$ (Gaussian init, std $1/\sqrt{r}$) and $B \in \mathbb{R}^{d_\text{out} \times r}$ (zero init), so $\Delta W = BA = 0$ at step 0. We patch the **query, key, and value** projections in all 12 encoder layers (36 adapters total). The paper uses Q+V; Q+K+V is a strict superset.
- **Task & data.** SST-2 (Stanford Sentiment Treebank), 67,349 train / 872 validation. The public test split has hidden labels (`label == -1`), so we evaluate on validation, matching the paper's GLUE convention.
- **Training.** AdamW, batch size 16, max sequence length 512, 10 epochs, cross-entropy loss, learning rate $5 \times 10^{-4}$, linear warmup (6%) + linear decay schedule.
- **Ablation grid.** Joint sweep over rank $r \in \{2, 4, 6, 8\}$ and scaling factor $\alpha \in \{2, 4, 6, 8\}$.
- **Hardware.** Single NVIDIA T4 in Google Colab. Each $(r, \alpha)$ setting takes ~2 hours.

**Deviations from the paper:**
- Q+K+V instead of Q+V (superset of paper's recommended setup).
- 10 epochs instead of 30–60 due to compute budget.
- Full-FT baseline uses the paper's reported number rather than a self-trained run at matched epochs (flagged as a limitation).
- Single seed instead of median over 5.

## 5. Reproduction Steps

1. Open `code/LoRA_RoBERTa_Classification.ipynb` in Google Colab and connect to a T4 GPU.
2. Mount your Google Drive — checkpoints and per-epoch CSVs write to `MyDrive/CS4782 Project/results/<timestamp>/`.
3. Set `rank` and `lora_alpha` in the hyperparameters cell to the configuration you want.
4. Run all cells. Expected wall-clock time: ~2 hours per $(r, \alpha)$ setting.
5. Outputs land in the timestamped results folder:
   - `LoRA_training.csv` — per-epoch training and validation loss
   - `LoRA_testing.csv` — final evaluation results
   - `LoRA_best_model.pth` — best checkpoint by validation loss

**Dependencies** (default-available on Colab): `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `pandas`, `scikit-learn`. No additional install required.

**Compute requirements.** Single T4 (16 GB VRAM) is sufficient. Peak VRAM usage ~4 GB.

## 6. Results / Insights

### Rank ablation on SST-2 ($\alpha = 8$ fixed)

| Method        | Trainable Params | % of RoBERTa | Val Accuracy |
|---------------|------------------|--------------|--------------|
| LoRA $r = 2$  | 0.70M            | 0.56%        | 55.1%        |
| LoRA $r = 4$  | 0.81M            | 0.65%        | 83.6%        |
| LoRA $r = 6$  | 0.92M            | 0.74%        | 94.6%        |
| LoRA $r = 8$  | 1.03M            | 0.82%        | 94.6%        |
| Full FT*      | 125M             | 100%         | 94.8%        |

*Full FT result taken from the paper (Hu et al., 2021, Table 2). The paper trains for 30–60 epochs; ours is a 10-epoch run, so this comparison is conservative.

See `results/figures/` for the full joint $(r, \alpha)$ sweep, including per-epoch training and validation curves.

### Key findings

1. **LoRA closes most of the accuracy gap with full fine-tuning** at $r \geq 6$ while training under 1% of parameters, validating the paper's core claim on our setup.
2. **Sharp rank threshold.** With $\alpha$ held constant at 8, accuracy drops off quickly below $r = 6$. Rank 4 trains but is unstable; rank 2 fails to train (~chance accuracy). The paper reports very low ranks ($r = 1, 2$) work on some tasks — we don't see that on SST-2 with our hyperparameters. We attribute this to the bottleneck width being too narrow at very low rank to express the SST-2 adaptation, despite the larger $\alpha/r$ scaling.
3. **The $\alpha/r$ ratio is the relevant knob.** Our joint sweep shows that runs with the same $\alpha/r$ ratio behave more similarly than runs with the same $r$ or same $\alpha$ alone — consistent with the paper's framing of $\alpha/r$ as an effective learning-rate scaling.
4. **Tiny parameter footprint.** At $r = 8$, only 1.03M of 125M parameters move. Frozen $W_0$ ships once and is reused across tasks; only $A$, $B$, and the classification head change.

## 7. Conclusion

LoRA reproduces well on a single-GPU setup with a smaller compute budget than the paper used. The most informative result for us was the joint rank/alpha sweep: the paper treats $\alpha$ as a fixed-and-forget convenience knob, but our results show that at constrained training budgets, the $\alpha/r$ ratio matters more than either parameter in isolation. This suggests the practical question "what rank should I use?" is more setup-dependent than the paper's headline numbers imply, and warrants checking on each new task rather than defaulting to a small $r$.

## 8. References

1. Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685 (2021).
2. Liu, Y. et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692 (2019).
3. Socher, R. et al. *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP 2013.
4. Wolf, T. et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP System Demos 2020.
5. Loshchilov, I. & Hutter, F. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019.

## 9. Acknowledgements

This work was completed as the final project for CS 4782 (Deep Learning) at Cornell University, Spring 2026. Thanks to the course staff for guidance on project scope and evaluation, and to HuggingFace for the `transformers` and `datasets` libraries that made this implementation tractable on a single GPU.
