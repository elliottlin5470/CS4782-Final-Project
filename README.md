# LoRA: Low-Rank Adaptation on RoBERTa-base / SST-2

Re-implementation of **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) for CS 4782 (Cornell, Spring 2026).

**Authors:** Elliott Lin, Anne Xia, Trisha Saini, Samarth Shridhar.

## 1. Introduction

LoRA freezes a pretrained Transformer's weights $W_0$ and learns a low-rank additive update $\Delta W = BA$ with rank $r \ll d$, so the forward pass becomes $h = W_0 x + (\alpha/r)\, BAx$. Only $A$ and $B$ are trained; $W_0$ stays frozen. The paper's main claim is that this matches or beats full fine-tuning on GLUE with orders of magnitude fewer trainable parameters and no added inference latency (since $BA$ can be folded back into $W_0$ at deployment).

This repo reproduces that claim on **RoBERTa-base / SST-2** and adds a joint sweep over rank $r$ and scaling factor $\alpha$.

## 2. Chosen Result

We reproduce the RoBERTa-base row of **Table 2** from the paper on the SST-2 sentiment classification task. The paper reports:

- Full fine-tuning: **94.8%** val accuracy, 125M trainable params
- LoRA $r = 8$: **95.1%** val accuracy, 0.3M trainable params

We extend the paper's experiments with a joint sweep over $r \in \{2, 4, 6, 8\}$ and $\alpha \in \{2, 4, 6, 8\}$ to investigate how the $\alpha/r$ ratio affects training stability — a hyperparameter interaction the paper does not analyze in detail.

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
│   └── group_lora_2page_report.pdf
├── LICENSE                                  # MIT
├── .gitignore
└── README.md
```

## 4. Re-implementation Details

- **Model.** RoBERTa-base (125M params) from HuggingFace `FacebookAI/roberta-base`.
- **LoRA adapter.** A `LoRA_Adapter` class wraps each `nn.Linear` and adds trainable matrices $A \in \mathbb{R}^{r \times d_\text{in}}$ (Gaussian init, std $1/\sqrt{r}$) and $B \in \mathbb{R}^{d_\text{out} \times r}$ (zero init), so $\Delta W = BA = 0$ at step 0. We patch the **query, key, and value** projections in all 12 encoder layers (36 adapters total). The paper uses Q+V; Q+K+V is a strict superset.
- **Task & data.** SST-2 (Stanford Sentiment Treebank), 67,349 train / 872 validation. The public test split has hidden labels (`label == -1`), so we evaluate on validation, matching the paper's GLUE convention.
- **Training.** AdamW optimizer, cross-entropy loss, batch size 16, max sequence length 512, 10 epochs, learning rate $5 \times 10^{-4}$, RoBERTa-base tokenizer.
- **Ablation grid.** Joint sweep over rank $r \in \{2, 4, 6, 8\}$ and scaling factor $\alpha \in \{2, 4, 6, 8\}$.
- **Hardware.** Single NVIDIA A100 in Google Colab.

**Deviations from the paper:**
- Q+K+V instead of Q+V (superset of paper's recommended setup).
- 10 epochs instead of 30–60 due to compute budget.
- Full-FT baseline uses the paper's reported number rather than a self-trained run at matched epochs.
- Single seed instead of median over 5.

## 5. Reproduction Steps

1. Open `code/LoRA_RoBERTa_Classification.ipynb` in Google Colab and connect to an A100 GPU.
2. Mount your Google Drive — checkpoints and per-epoch CSVs write to `MyDrive/CS4782 Project/results/<timestamp>/`.
3. Set `rank` and `lora_alpha` in the hyperparameters cell to the configuration you want.
4. Run all cells. Each $(r, \alpha)$ setting takes ~1–2 hours.
5. Outputs land in the timestamped results folder:
   - `LoRA_training.csv` — per-epoch training and validation loss
   - `LoRA_testing.csv` — final evaluation results
   - `LoRA_best_model.pth` — best checkpoint by validation loss

**Dependencies** (default-available on Colab): `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `pandas`, `scikit-learn`. No additional install required.

## 6. Results / Insights

Our best $r = 8$, $\alpha = 8$ run reached **90.9% test accuracy** on SST-2, compared to the paper's reported 95.1%. We attribute the gap primarily to training for 10 epochs rather than 30–60 — at the end of epoch 10, the training loss curve was still consistently decreasing.

### Key findings

1. **LoRA gets close to full fine-tuning at $r \geq 6$** while training under 1% of parameters, supporting the paper's core claim despite the shortened training schedule.

2. **The $\alpha / r$ scaling factor is more important than the paper suggests.** When we held $\alpha = 8$ fixed and swept rank, $r = 2$ and $r = 4$ failed to train. We initially read this as a rank threshold. It is actually an $\alpha/r$ effect: at $r = 2$, $\alpha/r = 4$, which makes the low-rank update too large relative to the frozen pretrained weights.

3. **Setting $\alpha = r$ recovers low-rank performance — and improves it.** Retraining $r = 2, 4, 6$ with matched $\alpha$ not only fixed the failures but *improved* test accuracy below $r = 8$. We suspect that for binary classification, a tighter bottleneck forces the model to discard noise and focus on the discriminative signal.

4. **Maximum stable $\alpha$ is roughly $2r$.** Holding $r = 2$ fixed and sweeping $\alpha \in \{2, 4, 6, 8\}$, performance is stable at $\alpha \leq 4$ and degrades sharply above that. This gives a concrete practical rule: keep $\alpha/r \leq 2$.

5. **Tiny parameter footprint.** At $r = 8$, only ~1.03M of 125M parameters move. Frozen $W_0$ ships once and is reused across tasks; only $A$, $B$, and the classification head change.

## 7. Conclusion

LoRA reproduces well on a single-GPU setup with a smaller compute budget than the paper used. The most informative finding for us was the $\alpha/r$ interaction. The paper treats $\alpha$ as a fixed-and-forget convenience knob set equal to $r$, but our sweep shows the ratio actively determines whether training succeeds at low rank. This suggests the practical question "what rank should I use?" is inseparable from "what $\alpha$ should I use?" — and that defaulting to $\alpha = r$, or any rule that keeps $\alpha/r \leq 2$, is more important than the paper's framing implies.

## 8. References

1. Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.

## 9. Acknowledgements

This work was completed as the final project for CS 4782 (Deep Learning) at Cornell University, Spring 2026. Thanks to the course staff for guidance on project scope and evaluation, and to HuggingFace for the `transformers` and `datasets` libraries that made this implementation tractable on a single GPU.
