# SDPO Paper: Comprehensive Notes
# "Reinforcement Learning via Self-Distillation"
# arXiv: 2601.20802

> Source: reconstructed from SDPO codebase (`SDPO/verl/`), GitHub repo (`lasgroup/SDPO`),
> and experiment scripts. The arXiv PDF was inaccessible (HTTP 403) during fetch.

---

## 1. Title and Authors

**Title**: Reinforcement Learning via Self-Distillation
**Short name**: SDPO (Self-Distilled Policy Optimization)
**Authors**: Jonas Hübotter et al. (Hübotter, Lübeck, Behric, Baumann, Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin, Krause)
**Year**: 2026
**arXiv**: 2601.20802

```bibtex
@article{hubotter2026reinforcement,
  title = {Reinforcement Learning via Self-Distillation},
  author = {Hübotter et al.},
  year = {2026},
  journal = {arXiv preprint arXiv:2601.20802},
}
```

---

## 2. Abstract (reconstructed)

Standard RLVR (GRPO, PPO) learns only from a scalar outcome reward per attempt — a **severe credit-assignment bottleneck**. SDPO introduces:

- **RLRF** (Reinforcement Learning with Rich Feedback): a framework that converts tokenized environment feedback (runtime errors, verifier outputs) into dense learning signals
- **SDPO** (Self-Distilled Policy Optimization): the algorithm that treats the current model conditioned on feedback as a "self-teacher" and distills its feedback-informed next-token predictions back into the policy

Three operating regimes demonstrated:
1. **No rich feedback** (scalar reward only): SDPO outperforms GRPO within 1-hour and 5-hour wall-clock windows on reasoning benchmarks
2. **Rich environment feedback**: faster convergence via denser credit assignment (logit-level > token-level > sequence-level)
3. **Test-time self-distillation (TTT)**: iterative refinement at inference — model generates multiple solutions, reuses best as demonstration, solves hard coding problems that base model or multi-turn alone cannot

Key claim: **no external teacher or explicit reward model needed** — entirely self-referential.

---

## 3. Problem: Credit-Assignment Bottleneck in RLVR

Standard RLVR gets only a binary scalar reward at the sequence level. The model knows a trajectory was wrong but not which tokens caused the error. Environments (code executors, math verifiers) routinely produce rich textual feedback explaining failures — SDPO converts this to dense per-token learning signal.

---

## 4. Method: SDPO Algorithm

### 4.1 Core Concept: Self-Teacher

Given student policy π_θ and a failed trajectory (prompt x, response y, reward 0):
1. Get environment feedback f ("Your answer is incorrect. The correct answer is 12.8." or runtime error)
2. A sibling rollout in the same batch that succeeded provides correct solution s
3. Construct reprompted context: `[original prompt] + [Correct solution: s] + [Feedback: f] + "Correctly solve the original question."`
4. Evaluate student policy conditioned on this enriched context: `π_θ(· | augmented_context)` — this is the **self-teacher**

**Critical**: the self-teacher does NOT generate a new response. It takes the original failed response tokens y and computes `p_teacher(token_t | augmented_context, tokens_{1..t-1})` for every token — same tokens, rescored under enriched context.

Verified in code (`ray_trainer.py:762`):
```python
teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
```

### 4.2 EMA Teacher Regularization

To prevent teacher collapsing to student (circular reasoning), an EMA of past student weights maintains a stable target:

```
teacher_param = (1 - τ) × teacher_param + τ × student_param
```

Where τ = `teacher_update_rate` (default 0.05; rich-feedback uses 0.01).

Code (`dp_actor.py:151`):
```python
teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
```

The teacher IS the reference model (same FSDP module), updated in-place. Lives CPU-offloaded, loaded shard-by-shard to GPU during teacher forward. Three model slots total (not four): actor (GPU), ref/EMA-teacher (CPU), vLLM rollout engine (GPU).

### 4.3 Alternative: Trust-Region Teacher

Interpolates between ref and student at forward time:
```
logits_teacher = lerp(logits_ref, logits_student, mix_coef)
```
Where `mix_coef = teacher_update_rate`. Keeps teacher as a mixture rather than updating weights.

### 4.4 Distillation Loss

Trains student to match self-teacher's distribution on reprompted context.

#### Full-logit formulation (`full_logit_distillation=True`)

For each response token position t, compute KL or JSD between student and teacher full vocabulary distributions.

**alpha = 0.0** (pure forward KL, teacher → student, coverage-seeking):
```
L = KL(p_student || p_teacher) = Σ_v p_student(v) log(p_student(v)/p_teacher(v))
```

**alpha = 1.0** (pure reverse KL, student → teacher, mode-seeking):
```
L = KL(p_teacher || p_student) = Σ_v p_teacher(v) log(p_teacher(v)/p_student(v))
```

**0 < alpha < 1** (generalized JSD):
```
M = alpha × p_teacher + (1 - alpha) × p_student   [mixture distribution]
L = alpha × KL(p_teacher || M) + (1 - alpha) × KL(p_student || M)
```

In log-space (numerically stable, from `core_algos.py:1138-1160`):
```
log M = logsumexp(log p_student + log(1-alpha), log p_teacher + log(alpha))
kl_teacher = KL(M || p_teacher)
kl_student = KL(M || p_student)
L = (1-alpha) × kl_student + alpha × kl_teacher
```

**alpha = 0.5** = symmetric JSD (paper Section 3 default, generalization experiments).
**alpha = 1.0** = reverse KL, mode-seeking (rich-feedback experiment).

#### Top-k approximation (`distillation_topk`)

Instead of full ~150k vocabulary softmax, only top-k tokens used. A "tail bucket" accumulates remaining probability mass:
```
tail_log = log(1 - sum(p_i for i in top-k))
```
Top-k+1 distribution is renormalized, KL computed over it.
- Generalization: `distillation_topk=100`
- Rich feedback: `distillation_topk=20`

#### Importance Sampling correction (`is_clip`)

Teacher forward uses different context than rollout → off-policy distribution shift. Corrected with per-token IS weights:
```
ratio_t = exp(log π_θ(y_t | context) - log π_rollout(y_t | context))
ratio_t_clipped = min(ratio_t, is_clip)    [is_clip = 2.0]
per_token_loss_t = KL_loss_t × ratio_t_clipped
```

Code (`core_algos.py:1173-1176`):
```python
negative_approx_kl = (student_log_probs - old_log_probs).detach()
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
per_token_loss = per_token_loss * ratio
```

Additionally: `rollout_is=token`, threshold=2.0 for off-policy correction in GRPO advantage estimator.

#### Non-full-logit distillation (requires alpha=1.0)

When `full_logit_distillation=False`, only chosen-token log-prob available:
```
log_ratio = log π_student(y_t) - log π_teacher(y_t)
per_token_loss = stop_gradient(log_ratio) × log π_student(y_t)
```
REINFORCE-style estimate of reverse KL at the chosen token.

### 4.5 Reprompting Logic (from `ray_trainer.py:672-815`)

For each sample in a batch of n rollouts per prompt:

1. Collect `success_by_uid`: which rollouts in the group scored above `success_reward_threshold` (= 1.0 for binary reward)
2. For each sample i:
   - If `dont_reprompt_on_self_success=True` and sample i succeeded: skip distillation
   - Otherwise: pick a sibling success as demonstration (random from group)
3. Build reprompt text using templates:
   ```
   {original_prompt}
   Correct solution:
   {sibling_successful_response}
   The following is feedback from your unsuccessful earlier attempt:
   {environment_feedback}
   Correctly solve the original question.
   ```
4. Tokenize reprompt to max `max_reprompt_len` tokens (truncate `left` / `right` / `error`)
5. Concatenate: `teacher_input_ids = [reprompt_tokens] + [original_response_tokens]`
6. Teacher forward (no grad): computes `log p_teacher(y_t | reprompt + y_{<t})` for all t
7. Student forward: computes `log p_student(y_t | reprompt + y_{<t})` for all t
8. Compute JSD/KL loss, apply IS weights, aggregate with `loss_agg_mode="token-mean"`

`self_distillation_mask = 1` for samples with a solution or feedback (reprompting occurred), 0 otherwise. Loss computed only on masked samples.

### 4.6 Training Objective

SDPO mode (`loss_mode=sdpo`) replaces the vanilla PPO policy loss with the distillation loss. The GRPO advantage computation still used to weight/filter rollouts (`adv_estimator=grpo` controls advantage estimation, not the policy loss).

---

## 5. Algorithm Pseudocode

```
Input: student policy π_θ, EMA teacher π_teacher (initially = π_ref),
       dataset D, hyperparams τ, α, topk, is_clip

For each training step:
  1. ROLLOUT: Sample n responses per prompt using π_θ (via vLLM)
     - Compute rollout log-probs: log π_rollout(y|x) per token [calculate_log_probs=True]

  2. SCORE: Evaluate each response r(y, ground_truth) ∈ {0, 1}
     - Extract textual feedback f(y, ground_truth) for wrong responses [if rich feedback]

  3. GRPO ADVANTAGE: For each prompt group g:
     - A_i = r_i - mean(scores_g)  [no std norm when norm_adv_by_std=False]

  4. BUILD SELF-DISTILLATION BATCH:
     - For each sample i with r_i = 0 (failed):
       - Find sibling success s_j (r_j = 1, same prompt)
       - Build reprompt: [x] + [solution: y_j] + [feedback: f_i] + "Correctly solve..."
       - Tokenize reprompt (truncate if > max_reprompt_len)
       - teacher_input_i = [reprompt_tokens] + [y_i_tokens]

  5. TEACHER FORWARD (no gradient):
     - Compute log π_teacher(y_i,t | teacher_input_i, y_i,<t) for all t
     - Extract top-k indices and log-probs

  6. STUDENT FORWARD (with gradient):
     - Compute log π_θ(y_i,t | teacher_input_i, y_i,<t) for all t
     - Extract top-k log-probs at same indices as teacher

  7. SDPO LOSS:
     - M = α × p_teacher + (1-α) × p_student
     - per_token_KL = JSD(p_student, p_teacher)
     - IS weight: ρ_t = min(exp(log π_θ(t) - log π_rollout(t)), is_clip)
     - per_token_loss = per_token_KL × ρ_t × self_distillation_mask_i
     - loss = mean over valid tokens

  8. BACKWARD + OPTIMIZER STEP

  9. EMA UPDATE:
     - π_teacher = (1 - τ) × π_teacher + τ × π_θ  [parameter space, CPU]
```

---

## 6. Experiments

### 6.1 Datasets and Benchmarks

**Generalization (no rich feedback)**:
- `datasets/sciknoweval/biology/`
- `datasets/sciknoweval/chemistry/`
- `datasets/sciknoweval/material/`
- `datasets/sciknoweval/physics/`
- `datasets/tooluse`

**Rich feedback**:
- `datasets/lcb_v6` — LiveCodeBench v6 (coding problems; code executor provides runtime errors as feedback)

**Test-time self-distillation (TTT)**:
- `lcb_v6_singles/q_{id}` — 19 hard LiveCodeBench problems
- Question IDs: 1, 3, 10, 43, 46, 59, 69, 74, 86, 91, 92, 95, 100, 103, 111, 120, 125, 127, 129

### 6.2 Models

- **Qwen/Qwen3-8B** (primary)
- **allenai/Olmo-3-7B-Instruct** (comparison in generalization)

### 6.3 Baselines

- **GRPO** — standard RLVR baseline
- **Multi-turn interaction** — iterative re-prompting without self-distillation (TTT comparison)
- **Base model** — no training

### 6.4 Results Summary

| Setting | Result |
|---|---|
| Generalization (no rich feedback) | SDPO > GRPO within 1h and 5h wall-clock on SciKnowEval chemistry and others. Metric: highest avg@16. |
| Rich feedback (coding, lcb_v6) | SDPO converges faster than GRPO; consistently outperforms when runtime error feedback is available |
| Test-time distillation (TTT) | Solves hard lcb_v6 problems that neither base model nor multi-turn interaction could solve |

### 6.5 Hardware

CSCS cluster (Swiss National Supercomputing Centre), NVIDIA GH200 GPUs, 4 GPUs per node, 288 CPUs/task, 460GB RAM.

---

## 7. Hyperparameters

### Generalization experiment (no feedback, `run_sdpo_all.sh`)

| Parameter | Value |
|---|---|
| model | Qwen/Qwen3-8B, allenai/Olmo-3-7B-Instruct |
| data.train_batch_size | 32 |
| rollout.n | 8 |
| actor.optim.lr | 1e-5 |
| actor.ppo_mini_batch_size | 32 |
| self_distillation.distillation_topk | 100 |
| self_distillation.alpha | 0.5 (JSD) |
| self_distillation.dont_reprompt_on_self_success | True |
| self_distillation.include_environment_feedback | False |
| optim.lr_warmup_steps | 10 |
| rollout.val_kwargs.n | 16 |
| algorithm.rollout_correction.rollout_is | token |
| gpus_per_node | 4 |
| time_limit | 12:00:00 |

### Rich feedback experiment (coding, `run_sdpo.sh`)

| Parameter | Value |
|---|---|
| model | Qwen/Qwen3-8B |
| data | lcb_v6 |
| data.train_batch_size | 32 |
| rollout.n | 8 |
| actor.optim.lr | 1e-6 |
| actor.ppo_mini_batch_size | 1 |
| self_distillation.distillation_topk | 20 |
| self_distillation.alpha | 1.0 (reverse KL) |
| self_distillation.dont_reprompt_on_self_success | True |
| self_distillation.teacher_update_rate | 0.01 |
| optim.lr_warmup_steps | 0 |
| rollout.val_kwargs.n | 4 |
| algorithm.rollout_correction.rollout_is | token |
| gpus_per_node | 4 |

### SDPO config defaults (`sdpo.yaml`)

| Parameter | Value |
|---|---|
| max_model_len | 18944 (rich feedback: prompt 2048 + feedback 8192 + response 8192 + template 512) |
| actor.ppo_mini_batch_size | 32 |
| policy_loss.loss_mode | sdpo |
| rollout.n | 8 |
| rollout.calculate_log_probs | True |
| algorithm.adv_estimator | grpo |
| algorithm.norm_adv_by_std_in_grpo | False |
| algorithm.rollout_correction.rollout_is | token |
| algorithm.rollout_correction.rollout_is_threshold | 2.0 |
| data.train_batch_size | 32 |
| actor.optim.lr | 1e-5 |
| trainer.val_before_train | False |

### actor.yaml self-distillation defaults

| Key | Default |
|---|---|
| policy_loss.loss_mode | "vanilla" (must override to "sdpo") |
| self_distillation.full_logit_distillation | True |
| self_distillation.distillation_topk | 100 |
| self_distillation.distillation_add_tail | True |
| self_distillation.alpha | 0.5 (JSD) |
| self_distillation.success_reward_threshold | 0.5 (YAML; use 1.0 for binary reward) |
| self_distillation.teacher_regularization | "ema" |
| self_distillation.teacher_update_rate | 0.05 |
| self_distillation.max_reprompt_len | 10240 |
| self_distillation.reprompt_truncation | "right" |
| self_distillation.dont_reprompt_on_self_success | True |
| self_distillation.remove_thinking_from_demonstration | True |
| self_distillation.include_environment_feedback | True |
| self_distillation.environment_feedback_only_without_solution | True |
| self_distillation.is_clip | 2 |
| ppo_mini_batch_size | 256 |
| use_kl_loss | false |
| entropy_coeff | 0 |
| optim.lr | 1e-6 |
| loss_agg_mode | "token-mean" |

---

## 8. Reprompt Templates (actor.yaml defaults)

```
reprompt_template:
  {prompt}{solution}{feedback}
  Correctly solve the original question.

solution_template:
  Correct solution:
  {successful_previous_attempt}

feedback_template:
  The following is feedback from your unsuccessful earlier attempt:
  {feedback_raw}
```

- `include_environment_feedback=False` (condition A): only `{prompt}{solution}` (feedback section empty)
- `include_environment_feedback=True` (condition D): both solution and feedback present
- No sibling success and no feedback: just original prompt (no reprompting)

**Custom no-feedback template** (used in our ema_teacher_jsd, ema_rkl_nofb, ema_jsd_wide scripts):
```
"{prompt}\n\nCorrect solution: {solution}\n\nCorrectly solve the original question."
```
Has NO `{feedback}` slot — using `include_environment_feedback=true` with this template would silently drop verifier text.

---

## 9. Key Design Decisions and Findings

| Decision | Detail |
|---|---|
| No external teacher | Model uses own weights conditioned on feedback as teacher — purely self-referential |
| Credit assignment hierarchy | logit-level (full distribution KL) > token-level (chosen log-prob) > sequence-level (scalar). SDPO at logit level. |
| JSD vs reverse KL | alpha=0.5 (JSD) for generalization / no-feedback; alpha=1.0 (rKL, mode-seeking) for rich feedback |
| norm_adv_by_std=False | sdpo.yaml default — unnormalized advantages preserve absolute reward scale; matches Dr.GRPO (arXiv:2503.20783) |
| Teacher does not generate | Only rescores existing tokens under enriched context — same tokens, different context → different per-token weights |
| Teacher update rate | Rich feedback: τ=0.01 (slower EMA, more stable); generalization: τ=0.05 (faster adaptation) |
| ppo_mini_batch_size=1 | Rich feedback experiment only — to fit large teacher input sequences (~18k tokens) |

---

## 10. Three Operating Conditions (Our Ablations)

| Condition | Script | alpha | feedback | Teacher context |
|---|---|---|---|---|
| A (generalization) | `ema_teacher_jsd` | 0.5 (JSD) | false | question + sibling solution |
| rKL no-feedback | `ema_rkl_nofb` | 1.0 (rKL) | false | question + sibling solution |
| D (rich feedback) | `ema_rkl_rich` | 1.0 (rKL) | true | question + sibling solution + verifier text |

---

## 11. Implementation Notes

**Entry point**: `python -m verl.trainer.main_ppo --config-name ppo_trainer`

**SDPO validation** (`main_ppo.py`):
- Rejects `use_kl_loss=True`: "SDPO cannot share the reference policy with KL regularization." (teacher IS the ref model; KL against ref would be circular)
- Requires legacy worker: "SDPO requires the legacy worker implementation to colocate the teacher."

**Worker colocalization**: When SDPO active, uses `Role.ActorRolloutRef` to colocate teacher policy with actor rollout worker. Teacher (= EMA of ref model) lives on same process as actor.

**"Async" clarification**: `async_rollout_mode=True` means agent loop workers handle prompts concurrently. Does NOT overlap rollout N+1 with training step N.

**Diagnostics logged per step**:
- `self_distillation/success_group_fraction`: fraction of groups with ≥1 success
- `self_distillation/success_sample_fraction`: fraction of samples with sibling solution
- `self_distillation/feedback_available_fraction`: fraction with non-empty feedback
- `self_distillation/feedback_used_fraction`: fraction where feedback entered reprompt
- `self_distillation/reprompt_sample_fraction`: fraction receiving reprompted teacher context
- `self_distillation/teacher_prompt_len_mean`: average reprompt length in tokens
- `self_distillation/teacher_prompt_truncated_fraction`: fraction hitting max_reprompt_len
- `self_distillation/teacher_total_len_mean`: average total teacher input length (reprompt + response)

**Per-sample JSONL fields** (added by our patch to `ray_trainer.py`):
- `reprompt_len`: teacher prompt token count (== max_reprompt_len if truncated)
- `feedback_used`: whether verifier feedback text was injected
- `has_solution`: whether a sibling correct solution was available

---

## 12. Related Work (referenced in codebase)

- GRPO (Group Relative Policy Optimization)
- RLOO (arXiv:2402.14740)
- REINFORCE++ (arXiv:2501.03262)
- ReMax (arXiv:2310.10505)
- Dr.GRPO (arXiv:2503.20783) — unnormalized advantages
- GSPO (arXiv:2507.18071)
- SAPO (arXiv:2511.20347)
- GAE (arXiv:1506.02438)
- Dual-clip PPO (arXiv:1912.09729)
- GPG (arXiv:2503.19595)
- OTB (Optimal Token Baseline)
- AdaptiveKL (arXiv:1909.08593)

---

## 13. max_model_len Breakdown (sdpo.yaml comment)

```
MAX_PROMPT_LENGTH  =  2048
MAX_RESPONSE_LENGTH = 8192
MAX_FEEDBACK_LENGTH = 8192
TEMPLATE_LENGTH    =   512  (heuristic upper bound, not enforced)
MAX_MODEL_LEN      = 18944  (= 512 + 2048 + 8192 + 8192)
```

For math tasks (our use case): `max_model_len=4096` (Qwen2.5-Math-1.5B max_position_embeddings).
- `max_response_length=3072`
- `max_reprompt_len=1024`  (= 4096 - 3072; question is INSIDE reprompt, not subtracted again)

---

## 14. Test-Time Self-Distillation (TTT)

Inference-only variant — no further training. At inference time:
1. Generate multiple candidate solutions
2. Use best candidate as demonstration for next attempt
3. Repeat iteratively

Solves 19 hard LiveCodeBench questions (IDs: 1, 3, 10, 43, 46, 59, 69, 74, 86, 91, 92, 95, 100, 103, 111, 120, 125, 127, 129) that neither the base model nor standard multi-turn interaction could solve.

---

*Last updated: 2026-05-14. Source: SDPO codebase + GitHub, arXiv PDF inaccessible.*
