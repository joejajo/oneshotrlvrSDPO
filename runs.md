# SDPO Training Runs

Baseline: Qwen2.5-Math-1.5B, MATH-500 greedy pass@1 = **32.6%**
Target: **>50%**

---

## Failed Runs (all used reprompt_truncation=left)

| Job | Script | τ | LR | Batch | Temp | GPUs | Result |
|-----|--------|---|----|-------|------|------|--------|
| 1578488 | nofeedback | 0.05 | 1e-5 | 32 | 1.0 | 8 | collapsed step 10 → 6.8%, never recovered above 32.6% by step 80 |
| 1578578 | nofeedback | 0.05 | 1e-5 | 32 | 1.0 | 2 | collapsed step 10 → 5.6%, partial recovery to 37.8% at step 18 |

**Root causes identified:**
1. `reprompt_truncation=left` stripped the problem statement from reprompt context → teacher computed distillation targets on corrupt sequences → student learned garbage token patterns
2. Possibly: LR=1e-5 too fast, batch=32 too small for stable GRPO advantage normalization

---

## Active Runs

### Run A — τ=0.001, clean reprompt (conservative anchor)
**Script**: `train_oneshot_sdpo_2gpu_stable.slurm`
**Output**: `output_2gpu_stable/`
**GPUs**: 2× A100

| Parameter | Value |
|-----------|-------|
| teacher_update_rate | **0.001** |
| reprompt_truncation | **right** |
| max_reprompt_len | **1024** |
| lr | 1e-6 |
| batch | 128 |
| temp | 0.6 |
| α | 0.5 |
| steps | 250 |

**Hypothesis**: τ=0.001 keeps teacher ≈ original model for all 250 steps (22% drift total), providing stable distillation anchor. Prevents catastrophic forgetting. Tradeoff: teacher learns π₁ very slowly, so reprompt demonstrations are weaker early on.

**Expected**: stable (no collapse), gradual improvement to ~38-45%.

---

### Run B — τ=0.001, clean reprompt, 4 GPUs
**Script**: `train_oneshot_sdpo_4gpu.slurm`
**Output**: `output_4gpu/`
**GPUs**: 4× A100

Same as Run A but 4 GPUs → faster FSDP update step, same learning dynamics.

**Expected**: same trajectory as Run A but faster per-step wall time.

---

### Run C — τ=0.05 (paper setting) + clean reprompt
**Script**: `train_oneshot_sdpo_tau05.slurm`
**Output**: `output_tau05/`
**GPUs**: 2× A100

| Parameter | Value |
|-----------|-------|
| teacher_update_rate | **0.05** (paper default) |
| reprompt_truncation | **right** |
| max_reprompt_len | **1024** |
| lr | 1e-6 |
| batch | 128 |
| temp | 0.6 |
| α | 0.5 |
| steps | 250 |

**Hypothesis**: The previous τ=0.05 runs collapsed because of corrupted distillation targets (left truncation), NOT because the teacher drifted too fast. With clean reprompts, τ=0.05 should work as the paper intends — teacher specializes on π₁ and provides strong demonstrations, GRPO reward signal drives generalization (as proven by One-Shot-RLVR).

**Expected if hypothesis correct**: stable, strong improvement, potentially >50%.
**Expected if hypothesis wrong**: collapse at step 10-20 again → confirms teacher drift rate is the problem.

**Key diagnostic**: watch `self_distillation/success_group_fraction` — if it stays high (≥0.5) without validation collapsing, the clean reprompt was the fix.

---

## What Each Run Isolates

```
Failed runs:  τ=0.05  +  LR=1e-5  +  left truncation  +  batch=32   → collapsed
Run A/B:      τ=0.001 +  LR=1e-6  +  right truncation  +  batch=128  → ?
Run C:        τ=0.05  +  LR=1e-6  +  right truncation  +  batch=128  → ?
```

Comparing Run C vs failed runs: changes are truncation + LR + batch (τ unchanged).
Comparing Run A vs Run C: only τ differs (0.001 vs 0.05). Everything else identical.

If Run C is stable → truncation was the primary cause.
If Run C collapses but Run A is stable → τ is the primary cause.
If both stable → we have two working configs; pick the one with higher peak accuracy.

---

## Next Steps (pending results)

- If Run C collapses: try α=1.0 + τ=0.05 (distillation loss acts as implicit regularizer)
- If both stable but below 50%: try longer training (500 steps) or τ=0.01 (middle ground)
- If either exceeds 50%: that's the winning config
