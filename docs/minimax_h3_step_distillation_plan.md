# MiniMax H3 — few-step distillation LoRA: implementation plan

Working document. Not committed, by the same convention as
`minimax_h3_optimization_backlog.md`. Do not `git clean` this directory.

**Purpose.** A self-sufficient plan for building a step-distillation LoRA for MiniMax H3, of the
kind lightx2v's Lightning adapters provide for Wan. Written so a session with no prior context can
implement it from this file plus the repository.

**Status when written:** nothing below is implemented. Every number is arithmetic or a measurement
taken from the released BF16 checkpoint's schedule maths; no distillation training has been run.

---

## 1. Verdict

Feasible, and cheaper than the usual framing of distillation suggests.

| | value | basis |
|---|---|---|
| Step cost vs ordinary LoRA training | **~1.5x** | forward-count arithmetic, section 8 |
| Extra memory vs ordinary LoRA training | **~1 adapter** | only one pass carries gradients |
| New primitives required | **none** | everything needed is already in the repo |
| Implementation effort | **3–5 engineer-days** | section 7 |

Two properties make it cheap:

- **The three networks collapse onto one frozen base.** Teacher, student and EMA target differ only
  by which LoRA adapter is active. `LoRAModule.multiplier` is a plain attribute
  (`networks/lora.py:98`, used at `:173`), so switching costs nothing and needs no new machinery.
- **Only the student pass carries gradients.** Teacher and EMA passes run under `no_grad` and store
  no activations, so peak memory is barely above ordinary LoRA training.

A third property is specific to H3 and removes half the usual work: **the released checkpoint is
already guidance-distilled**. Lightning distils CFG and steps simultaneously; here only steps
remain, so the student needs no `w`-conditioning input and the distributional move is smaller.

---

## 2. Architecture facts this plan depends on

- 33.12B single-stream DiT, 50 blocks, hidden 5376, 56 heads x 128, ffn 14336, patch (1,2,2).
- Video and stereo audio are **packed into one sequence** with text and denoised jointly under full
  bidirectional attention.
- Flow matching with `x_t = (1 - sigma) * x0 + sigma * noise`, model predicts the data-pointing
  velocity `x0 - noise`, and `timestep = 1 - sigma`.
- **Two flow shifts**: `VIDEO_FLOW_SHIFT = 12.0`, `AUDIO_FLOW_SHIFT = 3.0`
  (`minimax_h3/architecture.py`). Both derive from one shared *unshifted* coordinate `u`.
- Exponential shift: `sigma = shift * u / (1 + (shift - 1) * u)`, inverted by `unshift_sigma`.
- adaLN is excluded from LoRA targets (`networks/lora_minimax_h3.py:31`) and must stay excluded:
  pruned inference builds replace it with a `[96768, r]` table, so an adaLN LoRA silently drops
  every key at inference.
- Sequence length is roughly 9.3k at 832x480x73 frames and 33k at 1280x704x124. Frame counts must
  satisfy `frame_count % 17 == 5`. Effective spatial stride is 32 px.

---

## 3. What already exists (do not rebuild)

| Capability | Where | Commit |
|---|---|---|
| Synchronized video/audio schedules from one shared `u` | `minimax_h3/training.py: prepare_joint_noisy_inputs` | `7675fde` |
| adaLN low-rank reduction, `--h3_adaln_rank` | `minimax_h3/adaln_lowrank.py`, `model_loader.py` | `1e8ec0a`, fp32 fix `74a6d61` |
| Two-pass training step (no_grad pass + grad pass) | `minimax_h3_train_network.py:586-647` | guidance distillation |
| Cross-modal training, observed-modality sigma pinning | `training.py: prepare_joint_noisy_inputs(observed=...)` | `e02651a` |
| Opt-in per-row timesteps | `packing.py: build_row_timesteps(per_row_timesteps=True)` | `71b731c` |
| Arbitrary condition placement, condition audio | `packing.py: build_t2va_packed_sequence` | `f4809b0` |
| Multi-step sampling loops | `minimax_h3/inference.py: denoise_fl2va:302`, `denoise_ref2va:426` | |
| Per-modality masked loss with weights | `training.py: joint_velocity_loss` | |
| **Adapter disable** | `networks/lora.py:703: set_enabled(bool)` | |
| **Frozen-base no_grad forward in the training step** | `minimax_h3_train_network.py:604-620` | `800b589` |
| **Student-vs-reference prediction loss** | `training.py:299: joint_prediction_loss` | `800b589` |
| Intermediate block feature capture via forward hooks | `minimax_h3/crepa.py:153-155` | `65338d1` |

**The base-preservation loss is the precedent to copy, and it already implements most of the
teacher machinery.** `--h3_base_preservation_loss_weight` disables the adapter with `set_enabled`,
runs one `no_grad` forward of the frozen base, and reduces the student against it through
`joint_prediction_loss`. That is exactly the teacher/student shape this plan needs.

Read `minimax_h3_train_network.py:604-620` before writing anything. In particular it wraps the
frozen branch in `torch.random.fork_rng(...)` so the trainable pass reuses the same stochastic
conditioning rows the frozen branch sampled. Any teacher pass added here must do the same, or the
two branches will silently disagree on their conditioning and the targets will be noise.

---

## 4. The maths

H3's probability flow is **linear** in this parameterisation, so the solver is exact and short.

With `x_t = (1 - s) * x0 + s * e` and the model predicting `v = x0 - e`:

```
x0      = x_t + s * v                     # recover the clean sample
e       = x_t - (1 - s) * v               # recover the noise
dx/ds   = e - x0 = -v
x_{s'}  = x_t - (s' - s) * v              # exact Euler step, any s -> s'
```

The last identity is exact for the true `v`, not a first-order approximation — verify by
substitution. All error therefore comes from `v` being model-predicted, never from the solver. A
Heun step (two evaluations, average the slopes) still helps because it evaluates `v` at both
endpoints.

**Consistency objective.** Choose a decreasing grid `u_0 = 1 > u_1 > ... > u_N = 0`. Let `f_theta`
be the student's implied `x0` prediction and `f_ema` the EMA copy. Train:

```
loss = d( f_theta(x_{u_n}, u_n),  f_ema(teacher_step(x_{u_n}, u_n -> u_{n+1}), u_{n+1}) )
```

with the EMA branch under `no_grad`. `d` is Huber (more stable than L2 for this objective).

**Per-modality application.** Everything above is applied *twice*, once per modality, with that
modality's own sigma:

```
s_video = shift(u, 12.0)
s_audio = shift(u,  3.0)
```

`prepare_joint_noisy_inputs` already does exactly this given `base_sigma=u`, so no change is needed
there.

---

## 5. Step grid — a measured finding that changes the recipe

**A uniform grid in `u` is unusable.** Measured on the released shifts:

```
N=8, u uniform:
  u        1.000  0.875  0.750  0.625  0.500  0.375  0.250  0.125  0.000
  s_video  1.000  0.988  0.973  0.952  0.923  0.878  0.800  0.632  0.000
  d_video  0.012  0.015  0.021  0.029  0.045  0.078  0.168  0.632
  s_audio  1.000  0.955  0.900  0.833  0.750  0.643  0.500  0.300  0.000
  d_audio  0.045  0.055  0.067  0.083  0.107  0.143  0.200  0.300
```

Seven video steps cover 0.368 of the schedule and the final step covers **0.632**. Nearly all
denoising would happen in one jump, which defeats the point of a multi-step student.

**Recommended grid: uniform in video sigma, recover `u` by `unshift_sigma`, derive audio from `u`.**

```
N=8, s_video uniform:
  s_video  1.000  0.875  0.750  0.625  0.500  0.375  0.250  0.125  0.000
  u        1.000  0.368  0.200  0.122  0.077  0.048  0.027  0.012  0.000
  s_audio  1.000  0.636  0.429  0.294  0.200  0.130  0.077  0.034  0.000
  d_video  0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125
  d_audio  0.364  0.208  0.134  0.094  0.070  0.054  0.042  0.034
```

Video is uniform by construction; audio is front-loaded but has no pathological step. This is the
default to implement.

**Why the grid must stay shared in `u`.** The two modalities attend to each other in one sequence,
and the released model was trained with both at the *same* schedule position. Building two
independent grids, one per modality, would present mismatched trajectory states to joint attention.
Always choose the grid in one space and derive the other by `unshift` then `shift`.

**Tuning knob if audio's first step proves too coarse:** interpolate between the two grids with a
power law on `u`, or place the grid uniformly in log-SNR. Expose the choice; do not hard-code.

---

## 5b. Choosing the teacher's field, and why no undistill adapter is needed

The released checkpoint is guidance-distilled, so its raw output is already a guided field at one
baked-in scale. Three teachers are available, in increasing cost:

| Teacher | Forwards | Guidance | Needs training? |
|---|---|---|---|
| **Distilled field** (model output as-is) | 1 | fixed at the released scale | no |
| **Guidance-recovered field** | 2 | any `w`, chosen per run | no |
| Base plus a CFG-undistill adapter | 2 | any `w` | yes, a separate adapter |

**The second option makes the third unnecessary in most cases.** `guidance_consistent_prediction`
in `minimax_h3/training.py` already inverts the distillation algebraically. With the model producing
`g = u + s * (c - u)` on the prompt and `g_empty ~= u` on the empty branch:

```
c_hat  = (g + (s - 1) * g_empty) / s        # raw conditional field
pred_w = g_empty + w * (c_hat - g_empty)    # re-apply guidance at any scale
```

Both terms come from ordinary forwards of the stock checkpoint, so a teacher with controllable
guidance costs one extra `no_grad` pass and no training at all. The trainer already wires the
two-branch evaluation under `--h3_guidance_distillation_scale`
(`minimax_h3_train_network.py:586-647`); reuse that code path rather than writing a new one.

**Where an undistill adapter would go, if one is ever trained.** On the **teacher only**. Teacher
and student do not have to share a base configuration -- the teacher merely manufactures targets.
This yields two deployment designs:

| | student trains on | inference loads | student must learn |
|---|---|---|---|
| **A, single adapter** | stock base | base + step LoRA | undistilled field *and* few-step |
| **B, stacked** | base + undistill adapter | base + undistill + step LoRA | few-step only |

Prefer **A**. One adapter, drops onto the stock checkpoint, and matches how few-step adapters are
consumed elsewhere. Its cost is that the student absorbs a larger delta, which may require more rank
than the 64 suggested in section 9. **B** asks less of the student but makes the result hostage to
the undistill adapter's quality and forces users to obtain two adapters.

Training the student on the stock base while its targets come from a guided teacher is not a
contradiction; it is the standard arrangement. The student learns to reach the guided trajectory in
few steps starting from unmodified weights.

**Recommended order.** Start with the distilled-field teacher: cheapest, and it tests the objective
without introducing a second variable. If samples look guidance-starved, or a different strength is
wanted, switch the teacher to the guidance-recovered field at a chosen `w` -- a flag, not a
redesign. Only consider an undistill adapter if the algebraic recovery proves inadequate, which
would mean `g_empty ~= u` is a poor approximation for this checkpoint. That is measurable directly:
compare `c_hat` reconstructed at one scale against the model's own output at another, and see
whether the identity holds across sigma.

Cost, in forward-equivalents per training step:

| | total |
|---|---|
| ordinary LoRA step | 4 |
| step-distill, distilled-field teacher | 6 (~1.5x) |
| step-distill, guidance-recovered teacher | 7 (~1.75x) |

---

## 6. Files to create and change

### New: `src/musubi_tuner/minimax_h3/distill.py`

```python
GridSpace = Literal["video_sigma", "base", "audio_sigma", "logsnr"]

def build_distillation_grid(
    num_steps: int,
    *,
    space: GridSpace = "video_sigma",
    video_shift: float = VIDEO_FLOW_SHIFT,
    audio_shift: float = AUDIO_FLOW_SHIFT,
) -> torch.Tensor:
    """Return the decreasing base-sigma grid `u` of length num_steps + 1, u[0]=1, u[-1]=0."""

def euler_step(sample, velocity, sigma_from, sigma_to) -> torch.Tensor:
    """x - (sigma_to - sigma_from) * v; exact for H3's linear flow."""

def heun_step(sample, velocity_from, velocity_to, sigma_from, sigma_to) -> torch.Tensor:
    """Average the endpoint slopes; two model evaluations."""

def predicted_x0(sample, velocity, sigma) -> torch.Tensor:
    """x + sigma * v."""

def consistency_loss(
    student_x0, target_x0, *, mask=None, delta: float = 0.001, form: Literal["huber","l2"]="huber"
) -> torch.Tensor:
    """Reduce one modality's consistency error."""
```

Keep this module free of trainer and model imports so it stays unit-testable on CPU.

### Already exists: adapter disable

Do **not** write a multiplier helper. `networks/lora.py:703` provides `set_enabled(bool)`, and
`minimax_h3_train_network.py:604-620` shows the established usage: unwrap the network through the
accelerator, assert `set_enabled` is callable, disable, run under `no_grad` inside
`torch.random.fork_rng(...)`, and re-enable in a `finally`. Reuse that block verbatim.

### New: EMA adapter

Clone the student's LoRA state dict; after each optimizer step do
`ema = decay * ema + (1 - decay) * student`. To evaluate the EMA branch, swap the tensors into the
network under `no_grad` (or keep a second network instance sharing the same frozen base — cheaper in
code, marginally more memory).

### Changed: `minimax_h3_train_network.py`

Add a distillation branch to `process_batch`, parallel to the existing guidance branch:

1. draw grid index `n` (uniform, or biased toward high sigma early in training)
2. `base_sigma = grid[n]` and `next_sigma = grid[n + 1]`
3. `inputs = prepare_joint_noisy_inputs(..., base_sigma=grid[n])`
4. teacher: disable the adapter with `set_enabled(False)` inside `torch.random.fork_rng(...)` and
   `torch.no_grad()`, predict, then `euler_step` per modality using **that modality's** sigma, and
   re-enable in a `finally` -- the shape of `minimax_h3_train_network.py:604-620`
5. student: predict with grad at `grid[n]` -> `predicted_x0`
6. EMA target: `no_grad` predict on the advanced sample at `grid[n + 1]` -> `predicted_x0`
7. loss: `consistency_loss` per modality, combined with the existing
   `--h3_video_loss_weight` / `--h3_audio_loss_weight` / `--h3_loss_balance` handling

Reuse `_predict` unchanged. Reuse the loss masks from the batch.

### CLI

```
--h3_distill_steps N                # student step budget, default 8
--h3_distill_grid {video_sigma,base,audio_sigma,logsnr}
--h3_distill_ema_decay              # default 0.95
--h3_distill_loss {huber,l2}        # default huber
--h3_distill_huber_delta            # default 0.001
--h3_distill_solver {euler,heun}    # default euler
--h3_distill_teacher {distilled,guidance_recovered}   # default distilled; see 5b
--h3_distill_teacher_guidance W     # only with guidance_recovered
```

Record all of them in `extra_metadata` so a produced adapter is self-describing.

---

## 7. Effort

| Task | Days |
|---|---|
| `distill.py` plus CPU unit tests | 1.5 |
| EMA adapter (adapter disable already exists, see section 6) | 0.5 |
| Trainer branch and CLI, modelled on the base-preservation block | 1 |
| Grid analysis, docs, metadata | 0.5 |
| Small-scale training bring-up and debugging | 1–2 |
| **Total** | **3–5** |

Lower than a first estimate of 5–8 because the teacher machinery -- adapter disable, the frozen
`no_grad` forward, RNG forking, and a student-vs-reference loss -- already ships with
`--h3_base_preservation_loss_weight`.

---

## 8. Cost and memory

**Throughput.** In forward-equivalents, with gradient checkpointing:

| | ordinary LoRA | distillation |
|---|---|---|
| student forward | 1 | 1 |
| checkpoint recompute | 1 | 1 |
| backward | 2 | 2 |
| teacher (`no_grad`) | — | 1 |
| EMA target (`no_grad`) | — | 1 |
| **total** | **4** | **6** |

**~1.5x an ordinary step.** A Heun solver adds one more teacher evaluation, and so does the
guidance-recovered teacher of section 5b; either takes it to ~1.75x, both together to ~2x.
Arithmetic, not measured — confirm on hardware before planning a long run.

**Memory.** Base weights are shared by all three roles. Only the student pass stores activations.
Additional cost over ordinary LoRA training is the EMA adapter plus, if a second network instance is
used, its module wrappers.

Base sizes: BF16 66.2 GB; `--h3_adaln_rank 16` 40.5 GB; `--fp8_base --h3_adaln_rank 16` **20.4 GB**.
Block-boundary activations at S ~ 9.3k are roughly 5 GB under gradient checkpointing.

- **80 GB card:** `--fp8_base --h3_adaln_rank 16` at 480p/73f fits with large headroom, no swapping.
- **24 GB card:** needs block swap (~10–20 blocks). Note the swap penalty now applies to 6
  forward-equivalents rather than 4. Backlog section L (online INT8 ConvRot composed with adaLN
  reduction, est. ~19 GB) would materially help here.

---

## 9. Hyperparameters to start from

| Parameter | Value | Note |
|---|---|---|
| LoRA rank / alpha | 64 / 32 | attn + mlp only; adaLN excluded |
| Learning rate | 5e-6 to 1e-5 | lower than ordinary LoRA |
| EMA decay | 0.95 | LCM's target-network value |
| Teacher grid points | 50–100 | discretisation the teacher steps on |
| Skip-k | 10–20 | teacher interval per training step |
| Student steps | 4–8 | the deliverable |
| Loss | Huber, delta ~1e-3 | scale with sqrt(feature dim) if unstable |
| Timestep sampling | uniform over grid indices | consider biasing to high sigma early |

No `w`-conditioning: the base is already guidance-distilled. Train with
`--h3_guidance_distillation_scale` **omitted** so the student learns the distilled field directly.

---

## 10. Tests

CPU, no checkpoint required:

1. `euler_step` reproduces the closed form: build `x_t` from known `x0`/`noise` at `s`, feed the
   *true* `v = x0 - noise`, assert the result equals the closed-form `x_{s'}` exactly.
2. `predicted_x0` recovers `x0` exactly from true `v`.
3. `build_distillation_grid` is strictly decreasing, starts at 1, ends at 0, has length `N + 1`.
4. Video-sigma grid gives uniform `d_video`; assert the pathological uniform-`u` case is *not* the
   default.
5. `set_enabled(False)` makes the output equal the base model's output, and the trainer re-enables
   the adapter even when the body raises.
6. EMA update arithmetic on a toy state dict.
7. `consistency_loss` masking matches `joint_velocity_loss` conventions.

GPU, small scale:

8. One distillation step runs forward and backward without error at 480p/73f.
9. Teacher pass produces bit-identical output to the base model with no adapter loaded.

---

## 11. How to know it works

Distillation quality cannot be read off the training loss. Plan for:

- **Fixed-seed sample grid** at 4, 8 and 50 steps, same prompt and seed, compared against the
  50-step base. The student at 8 steps should approach the base at 50.
- **Motion metric.** Few-step distillation *systematically reduces motion* — a documented VQ-up /
  MQ-down tradeoff across Lightning, CausVid and Self-Forcing. Measure it deliberately rather than
  discovering it later; adjacent-frame embedding distance is a cheap proxy.
- **AV sync.** Specific to H3 and the thing most likely to degrade, because audio and video traverse
  their schedules at different rates under any shared grid. Compare onset alignment between the
  student's audio and video against the base's.
- **Both modalities separately.** A student that is good at video and poor at audio will look fine
  on a video-only metric.

---

## 12. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Shared grid serves one modality poorly | **high, H3-specific** | section 5; make the grid space a flag and sweep it |
| Motion reduction | high, expected | measure from the first run; consider fewer distilled steps |
| Compute for a *real* run | high | small-scale proof first; a shippable adapter is cluster-scale |
| EMA/teacher divergence early in training | medium | warm up with a small skip-k, raise it later |
| LoRA multiplier not covering every module | medium | test 5 above |
| adaLN accidentally added to LoRA targets | medium | keep `lora_minimax_h3.py:31` exclusion; assert in a test |

**Explicitly out of scope.** Self-Forcing and CausVid-style causal distillation need causal
attention masking (`model.py` builds only a padding mask and exposes no topology argument), and the
Causal Forcing result says a causal student initialised from a *bidirectional* teacher violates
frame-level injectivity and collapses toward conditional expectation. Do not start there.

---

## 13. Environment

- Run GPU work on the shared box; check free VRAM first, it is shared.
- Use the `comfy` virtualenv; the `minimax-h3` one lacks `accelerate`.
- Released BF16 checkpoints live under the `Comfy-MiniMax-H3` model directory.
- Exact paths are recorded in section M of `minimax_h3_optimization_backlog.md`.
- Full test suite: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## 14. Suggested sequencing

1. **Grid analysis first, before any code.** Half a day. Reproduce section 5, sweep
   `N in {4, 6, 8}` across grid spaces, and decide the default. If no grid gives both modalities
   sane spacing, that is a finding that reshapes the plan.
2. `distill.py` plus its CPU tests. Purely mathematical, no model needed.
3. EMA adapter, with tests. The adapter-disable half already exists.
4. Trainer branch, copied from the base-preservation block at
   `minimax_h3_train_network.py:604-620`; get one step to run at 480p/73f.
5. Short run at low resolution; confirm the loss decreases and samples at 8 steps improve.
6. Only then consider scale.

---

## 15. Open questions

- Does a shared-`u` grid exist that serves both shifts well, or is a per-modality step count needed?
  Section 5 suggests uniform-video-sigma is adequate; unverified in training.
- Does the guidance-distilled base behave like a clean flow model under consistency distillation, or
  does the distilled field interact badly with the objective? No precedent found. Section 5b gives
  the fallback if it does: switch the teacher to the guidance-recovered field, which is a flag
  rather than new work.
- Is `g_empty ~= u` a good approximation for this checkpoint? Section 5b's recovery identity rests
  on it, and it is directly measurable without any training.
- Is rank 64 enough capacity for a step-distillation adapter on a 33B base, or does this need
  substantially higher rank than a style LoRA?
- Would `--h3_adaln_rank` change distillation quality, given the student sees a slightly different
  base than the published one? Section K of the backlog has the fidelity numbers.
