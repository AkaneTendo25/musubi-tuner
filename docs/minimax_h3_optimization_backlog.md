# MiniMax H3 — VRAM & speed optimization backlog

Working document. Consolidates every optimization identified while auditing the H3 training path
against (a) musubi-tuner's existing infrastructure, (b) other trainers, (c) H3's own architecture,
and (d) `github.com/shootthesound/Fizgig` (which trains the same 33B H3 checkpoint, image-only).

## Reading the columns

**Type** — `VRAM`, `Speed`, `Both`, `Quality`, or `Enabler` (unlocks other items).

**Evidence** —
- `verified` — arithmetic or code checked directly against this repo during the audit.
- `measured` — a number reported by another project on its own hardware. Directionally trustworthy,
  magnitude not guaranteed to transfer.
- `analysis` — derived from reading our code; not yet benchmarked.

**Effort** — `S` ≈ hours, `M` ≈ a day or two, `L` ≈ multi-day with real design risk.

**Risk** — chance of silently changing numerics or breaking a training run, not chance of failing to help.

## Workload calibration (read this before prioritizing)

Sequence length from `architecture.py`'s geometry. Effective spatial stride is **32 px** (VAE 16×
then DiT patch 2×2), and frame counts must satisfy `frame_count % 17 == 5` (`is_valid_frame_count`),
so only the buckets below are legal:

| Clip | Video latent frames | Patch grid | Video rows | `S` (incl. audio + text) | `[1,S,5376]` bf16 |
|---|---|---|---|---|---|
| 832×480, 73 frames | `((73-5)//17)*5+2 = 22` | 26 × 15 | 8,580 | **≈ 9.3k** | 96 MB |
| 1280×704, 107 frames | `((107-5)//17)*5+2 = 32` | 40 × 22 | 28,160 | **≈ 29k** | 297 MB |
| 1280×704, 124 frames | `((124-5)//17)*5+2 = 37` | 40 × 22 | 32,560 | **≈ 33k** | 344 MB |

Two corrections to earlier drafts of this document: `S ≈ 35k–150k` was wrong by roughly 4× and
inflated every memory-traffic estimate; and a "720p, 121 frames" example was **invalid geometry**
(121 % 17 = 2, so `temporal_shape(121)` raises). Use 107 or 124 frames, and 704 rather than 720
(720 is not a multiple of 32).

**The consequence that reorders the backlog: H3 training is compute-bound, not bandwidth-bound and
not PCIe-bound.** Rough FLOP budget at `S ≈ 29k`, per forward: linear layers ≈ 38.5 GFLOP/token ×
29k ≈ **1.1 PFLOP**; attention ≈ `4·S²·inner_dim` × 50 blocks ≈ **1.2 PFLOP**. A full step
(forward + checkpoint recompute + backward) is ≈ 4× forward ≈ 9 PFLOP, i.e. tens of seconds per
step on a single consumer card. Against that:

- Streaming all 50 blocks at FP8 moves ~32 GB over PCIe per forward ≈ 1.3 s at 25 GB/s — a few
  percent of step time. **Block swapping is already cheap at these sequence lengths.**
- Modulation-tensor traffic (A1) is ~200 GB per forward ≈ 0.13 s at 1.5 TB/s — **under 1%.**
- Per-step fixed costs (syncs, CPU geometry construction, H2D of latents) are single-digit
  milliseconds against a multi-second step — **well under 0.1% each.**

So the memory-traffic and sync items in section A are **VRAM wins with negligible speed effect**.
The genuine speed levers are the ones that reduce FLOPs or make them cheaper: attention backends
(B1), quantized matmul that actually computes in low precision (B5/D1/D4), and token dropping (G1).

This also tempers the borrowed `torch.compile` numbers: Fizgig measured ~2×, but on a 12.9B image
model at 1–4k tokens where launch overhead and elementwise work dominate. At H3's token counts
elementwise work is ~1% of the step, so expect **single-digit percent**, not 2×.

---

## A. Forward-pass waste (our own code, no new dependencies)

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| A1 | Segment-broadcast modulation instead of `index_select` expansion — `model.py:256-265` expands a 6-row table into six dense `[S,5376]` tensors per block, 300 full-sequence temporaries per forward, re-materialized in backward. Segments are contiguous by construction in `packing.py`. | VRAM | analysis | M | Low |
| A2 | **MOSTLY SUBSUMED by E5** — the 13 GB H2D saving becomes ~77 MB. Hoist `adaln_proj` out of the checkpointed region — `model.py:252` depends only on the timestep embedding but is recomputed in the backward of all 50 blocks. Under layer-granular streaming the backward *re-acquires and re-dequantizes* the weight: ~13 GB of extra H2D per step. | Both | analysis | M | Low |
| A3 | `index_select` before the output heads and cast to FP32 once — `model.py:289-292` casts the entire packed sequence to FP32 **twice** (video and audio dtypes are identical), projects all `S` rows through both heads, then discards text rows. | VRAM | analysis | S | Low |
| A4 | Fuse/in-place the RoPE application and drop the no-op `.contiguous()` — `model.py:80-87` allocates ~5 full-size temporaries per q and per k per block. | Both | analysis | M | Low |
| A5 | In-place `index_copy_` for packed-stream assembly — `model.py:440-443` uses three out-of-place calls, allocating 4 full-size buffers where 1 suffices, plus real FP32→BF16 copies of the whole video/audio blocks. | VRAM | analysis | S | Low |
| A6 | Remove the dead padding-mask sync — `model.py:449` `bool(is_padding.any())` forces a GPU→CPU sync every forward, and **no code path in this fork ever produces a negative tag** (`packing.py:201,222-225,381-384`). | Speed | verified | S | Low |
| A7 | Remove loss-path syncs — `training.py:162` `int(valid.sum().item())` and `:172` `masked_select` (data-dependent output size ⇒ implicit sync + full extra allocation). All-valid case should be a plain `.mean()`. | Speed | analysis | S | Low |
| A8 | Remove sigma-validation syncs — `training.py:46` runs `bool(...any())` on a device tensor, reached 3× per step via `prepare_joint_noisy_inputs` and `map_sigma_between_shifts`. | Speed | analysis | S | Low |
| A9 | Gate the per-step logging `float(...)` calls on tracker presence — `train_network.py:460-465` builds four unconditionally; `trainer_base.py:2147` only uses them when a tracker is attached. | Speed | analysis | S | Low |
| A10 | Drop the discarded noisy-latent computation — `train_network.py:416` calls `get_noisy_model_input_and_timesteps` for its timesteps only, computing and throwing away a full-size tensor that `prepare_joint_noisy_inputs` then recomputes. | Both | analysis | S | Low |
| A11 | Cache packing geometry, `position_ids` and RoPE `cos`/`sin` per shape bucket — `packing.py:189-239` rebuilds a float64 CPU `[S,3]` tensor, meshgrid, tags and three `arange`s every step; `model.py:434` rebuilds cos/sin every forward. All depend only on the shape bucket. | Speed | analysis | M | Low |
| A12 | Replace `torch.unique(row_timesteps, ...)` — `packing.py:427` sorts the full `S`-length vector every step to recover 2–4 distinct values. | Speed | analysis | S | Low |
| A13 | Reuse geometry across the guidance-distillation double pass — `integration.py:430-448` reruns the entire packing/layout/timestep pipeline twice per step with identical geometry. | Speed | analysis | S | Low |
| A14 | `pin_memory=True` on the DataLoader (`trainer_base.py:1650-1657`) plus `non_blocking` on the eight per-step `.to(device)` calls at `integration.py:488-495`. | Speed | analysis | S | Low |
| A15 | Stop checkpointing `token_refiner` — `model.py:438` checkpoints 2 blocks that contain **no trainable parameters** (excluded from LoRA targets), so the recompute buys nothing. | Speed | analysis | S | Low |
| A16 | Stop forcing `requires_grad` on video/audio inputs — `train_network.py:373-375` makes backward propagate through the frozen patch projections and refiner all the way to the latents. | Both | analysis | S | Med |
| A17 | Chunked SwiGLU over the sequence dim — `model.py:189` materializes `[1,S,28672]` (5.7 GB bf16 at S=100k) before chunking. | VRAM | analysis | M | Low |
| A18 | Replace the dense `[S,S]` bool mask with varlen `cu_seqlens` — `model.py:450` builds a 10 GB mask at S=100k and disqualifies SDPA's flash/mem-efficient kernels. Currently unreachable (see A6) but a landmine for any padded/ref2va path. | VRAM | analysis | M | Med |

## B. musubi machinery that exists but H3 rejects

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| B1 | Wire `modules/attention.py` so flash / sage / xformers work — `train_network.py:140-143` hard-rejects everything except `--sdpa`. H3's attention is maskless batch-1 with head_dim 128, the easiest possible case. Sage (INT8 QK) is typically ~2× on the attention portion, which dominates at large `S`. | Speed | analysis | M | Med |
| B2 | Enable `--gradient_checkpointing_cpu_offload` — `model.py:332-334` raises, though `utils/model_utils.py:226-248` is arch-agnostic. Block-boundary activations are 50 × `[1,S,5376]` ≈ 53 GB at S=100k; this is what makes long/HD clips fit at all. | VRAM | analysis | S | Low |
| B3 | Enable per-block `torch.compile` — rejected at `train_network.py:138` / `model.py:332`, though `disable_linear_from_compile()` already makes it block-swap-safe (used by wan/qwen/hv15). H3's forward is unusually elementwise-heavy (RoPE, 6-tensor modulation, SwiGLU, residual gates). | Speed | analysis | M | Med |
| B4 | Expose FP8 `quantization_mode` / `block_size`; borrow Kandinsky's swap-aware block-size selection (`kandinsky5_train_network.py:597-616`) and its graceful try/except fallback. | VRAM | analysis | S | Low |
| B5 | ~~Enable `use_scaled_mm` / `--fp8_fast`~~ — **withdrawn**. Implemented and then removed. The FP8 matmul takes one scale per operand, so it needs `--h3_fp8_quantization_mode tensor`, and per-tensor measured 5.06 s/it against per-block's 3.94 — a 1.28× deficit the ~1.13× matmul gain cannot repay. Upstream's shared forward is unusable anyway: its guard demands a rank-one weight scale while per-tensor quantization emits a rank-zero scalar, it casts activations to FP8 with no scale, and `torch._scaled_mm` carries no derivative. | Speed | measured | — | — |
| B6 | Use `--num_timestep_buckets` — already generic; compounds with A11's per-bucket geometry cache. | Speed | analysis | S | Low |

## C. Environment and toolchain (cheapest wins in the list)

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| C1 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Confirmed absent from our tree. Fizgig v2.8.2 traced per-step time **roughly doubling** two-thirds through a run — 90% util, full clocks, no throttling — to allocator fragmentation near a full card; this removed it entirely. We run near-full VRAM by design. | Speed | measured | S | Low |
| C2 | `KMP_BLOCKTIME=0` + `OMP_WAIT_POLICY=PASSIVE`. Absent from our tree. Fizgig measured "14.8 cores busy in what should be idle time → 0.0", and credits it with **up to ~1.6× step rate** because spinning threads competed with those launching GPU work. Interacts directly with our offloader's worker threads. | Speed | measured | S | Low |
| C3 | Raise `torch._dynamo.config.cache_size_limit` (default **8**) and `accumulated_cache_size_limit` (default 256). We expose a flag but default to PyTorch's value and never touch the accumulated limit. Past the limit dynamo **silently falls back to eager, permanently, with nothing in the logs** — Fizgig's first compile attempt measured *slower* (0.794 vs 0.610 s/it) for this reason. | Enabler | verified | S | Low |
| C4 | Put the gradient checkpoint **inside** the compiled region rather than wrapping outside. Fizgig measured **1.19× per block** (8.817 → 7.428 ms). Our H3 checkpoints outside (`model.py:401-413`). | Speed | measured | M | Med |
| C5 | Patch torch's sympy `Mod.eval`, which asserts on negative values that inductor's tiling substitutes — aborts compilation, then segfaults the process (exit 139). Wrap rather than vendor so it no-ops if torch fixes it. | Enabler | measured | S | Low |
| C6 | MSVC / `vcvars64.bat` auto-bootstrap on Windows so `cl.exe` is findable — removes the biggest local barrier to `torch.compile`. | Enabler | measured | S | Low |

## D. Quantization

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| D1 | **INT8 W8A8 training via `torch._int_mm`.** Frozen base ⇒ `grad_weight` is never needed, so a custom autograd Function returns `grad_input` only — and weight gradients are exactly where int8 hurts most. Gradient checkpointing runs the forward twice, so an int8 forward with a bf16 backward already covers ~2/3 of the matmul work with exact gradients. Measured (M=1024, K=N=6144, RTX 5090): 0.767 ms → **0.224 ms**; ~11% end-to-end vs NF4 with ~7× lower forward error; works from Turing, unlike `_scaled_mm` (sm_89+). | Both | measured | L | High |
| D2 | INT8 weight layout: store `(N,K)`-contiguous and pass `.t()` to `_int_mm`. The transpose is a free view and yields the column-major layout int8 tensor cores want — **0.131 ms vs 0.452 ms**. Pre-transposing "to avoid a transpose" is 3.4× *slower*. | Speed | measured | — | — |
| D3 | INT8 `M ≤ 16` zero-pad to 17 rows and slice back (`_int_mm` refuses M ≤ 16). Justification is **determinism**, not speed: otherwise whether a matmul runs quantized depends on token count and silently changes answers. | Quality | measured | — | — |
| D4 | FP8 `_scaled_mm` with per-token activation quantization and a **post-matmul** scale (pass `scale_a=1.0`, multiply after) — sidesteps Blackwell's rowwise-scale limitation. Claimed 1.3–1.5×. | Speed | measured | M | Med |
| D5 | Relax the FP8 `M % 16` alignment guard — only K and N need 16-alignment. Fizgig measured **1.89× at M=4173**. Our packed sequences are always ragged, so this would bite us. | Speed | measured | S | Low |
| D6 | NF4 base via bitsandbytes: 33 GB (FP8) → ~17 GB. We already have bnb plumbing from text-encoder caching. Note NF4 and block swap are **mutually incompatible** (packed data is a plain attribute the offloader can't see). | VRAM | measured | M | Med |
| D7 | **OBSOLETE — superseded by E5.** **Mixed quantization: FP8 for the LoRA-adapted attn/mlp Linears, NF4 for adaLN** (13 GB → 6.5 GB). adaLN is the best NF4 candidate in the model — it sees only ~6 distinct input rows per step on a 1-D manifold. Fizgig NF4s adaLN already; we FP8 it. | VRAM | analysis | M | Med |
| D8 | Checkpoint-safety discipline for any quantized matmul path (prerequisite for D1/D4): probe the fast-path decision **once and cache it per module** (a per-call try/except makes forward and recompute diverge ⇒ `CheckpointError`); stash frozen tensors on `ctx` directly, not `save_for_backward` (shifts the saved-tensor list); gate on `self.training`, not `is_grad_enabled()` (flips between forward and recompute); catch `RuntimeError`, never bare `Exception` (swallows checkpoint's private `_StopRecomputationError`). | Enabler | measured | — | — |
| D9 | If adding NF4: construct `Linear4bit` shells inside `with torch.device("meta")` or each constructor eagerly allocates a full fp32 CPU weight (~118 GB of throwaway across H3's targets); `nn.Module.to()` does **not** move `_nf4_packed`/`_nf4_state`; `QuantState.to()` is broken in bnb 0.48.x for the non-nested case. | Enabler | measured | — | — |

## E. H3-architecture-specific

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| E1 | **OBSOLETE — superseded by E5.** **CPU-resident adaLN.** `adaln_proj` produces ~6 rows × 96768 per block per step from inputs known before the forward begins: ~58 MB of output and ~80 GFLOP total per step. Compute it on CPU in a pipelined worker thread (timesteps are sampled at step start, so it overlaps the previous step's backward). 13 GB of weights then live permanently in system RAM and never touch VRAM or PCIe — versus streaming, which moves 13 GB H2D per forward (26 GB with the backward re-acquire). Composes with A2 and is quantization-format-independent. | VRAM | analysis | L | High |
| E2 | Fused modulate kernel (Triton): RMSNorm + row-gather + `(1+scale)·x + shift` in one memory pass per contiguous segment. Collapses A1 and the two-step modulate. `torch.compile` (B3) gets most of this for free. | Both | analysis | L | Med |
| E3 | Fused RoPE kernel exploiting the 96-of-128 rotated-dims split — avoids the cat/copy dance in A4 entirely. | Both | analysis | L | Med |
| E4 | **SUBSUMED by E5** (automatic once adaLN is reduced). Shrink the streaming ring: slots are sized to the largest payload, which is `adaln_proj` (~500 MB/slot at FP8). Pulling adaLN out of the streamed set (via A2/E1) shrinks ring memory ~2×. | VRAM | analysis | S | Low |

## F. Shape stability and scheduling

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| F1 | Round packed sequence lengths **up to a multiple of 64**. Trimming to the exact valid length sounds optimal and isn't: the length carries each caption's token count, so every shape-planning backend (cuDNN, compile's shape cache, cuBLAS heuristics) pays a first-sight cost it can never amortize. Fizgig measured 30 distinct shapes → **10 shapes for +3.6% tokens**. Applies directly to our variable-length packed AV sequences. | Speed | measured | M | Low |
| F2 | Resolve any shape/trim decision **once per forward**, not per block. Reading a CUDA tensor on the CPU inside each block was a device sync ×28 (×56 under checkpointing) **and a hard graph break — "why compiling the blocks lost end to end while the same block compiled 1.37× faster in isolation."** Same pattern as A6. | Speed | measured | S | Low |
| F3 | Set the SDPA backend as a **priority list, not a forced backend** — forcing cuDNN raises "No available kernel" on head_dim > 128 (e.g. single-head VAE attention). | Speed | measured | S | Low |
| F4 | cuDNN-attention cost-model auto-switch: ~1.3 s planning per distinct shape but ~6% faster steady-state; decide at an epoch boundary once every shape has been observed (`needed = n_shapes × 35 × 2.0`). | Speed | measured | M | Low |
| F5 | **DONE (7675fde).** **Switch the timestep base distribution from logit-normal to uniform.** We default `timestep_sampling="shift"` + `discrete_flow_shift=12.0` (`minimax_h3_train_network.py:547-551`). musubi's `shift` mode composes the shift onto a **logit-normal** base (`trainer_base.py:629-632`, `t = sigmoid(sigmoid_scale*randn())`). Both other known H3 implementations apply shift 12 to a **uniform** base instead. The composition is what collapses the low-noise tail: P(sigma<0.5) is 0.65% for us vs 7.70% for them. Fix: sample uniform, keep shift 12. | Quality | verified | S | Low |
| F6 | **PARTLY DONE (7675fde)** — shifts are now freely settable via `--h3_shift_video`/`--h3_shift_audio`; the token-aware shift is still open. Make `--discrete_flow_shift` freely settable (drop the warning at `:122-127`) once J2 is fixed, and add an H3-specific token-aware shift keyed on **target video rows** `T_latent*(H/32)*(W/32)` -- not packed `S`, which varies with caption and reference rows independently of video dimensionality. musubi's existing `flux_shift`/`qwen_shift` read only the last two latent dims, so they ignore temporal length entirely and cannot be reused as-is for H3's 5D latents. | Quality | verified | M | Med |

## G. Other trainers

| # | Optimization | Type | Evidence | Effort | Risk |
|---|---|---|---|---|---|
| G1 | TREAD-style token routing — drop a random fraction of video tokens through the middle blocks, reinserting via the residual. At large `S` × 50 blocks this is a 30–50% step-time cut for modest quality cost; H3's packed layout with explicit `video_indices` makes routing straightforward. | Speed | analysis | L | High |
| G2 | Stochastic-rounding bf16 optimizer (OneTrainer's `adamw_bf16`). The repo has none (`trainer_base.py:1702` disables `full_bf16` for exactly this reason). Low priority while LoRA-only, but it's the enabler for any full-bf16 path. | VRAM | analysis | M | Med |
| G3 | VRAM auto-planner: **probe** capability by running a tiny `_scaled_mm`/`_int_mm` rather than checking compute capability; read **free** VRAM fresh, not total, not cached; fit a linear peak model; pick the **fewest** swapped blocks that fit. Fizgig's ladder is *quantize harder before swapping at all* — heavy swap costs **4.4× the time and 4× the CPU**. | Both | measured | L | Low |
| G4 | `torch.compile` break-even auto-decision: measured warm-up ÷ measured per-step saving × 2.0 margin, with hard refusals (block swap active, no triton, no host C compiler). | Speed | measured | M | Low |
| G5 | 8-bit / paged optimizers and CAME — **already available** via the dotted-path optimizer loader. No work required. | VRAM | verified | — | — |

## H. Already correct — do not change

| # | Item | Note |
|---|---|---|
| H1 | FP8 quantization of adaLN | **REVERSED by E5.** Was: correct call. adaLN is 13.0B of 33.1B (39.4%); it sees only ~6 distinct input rows per step on a 1-D sigma manifold; the matmul accumulates over 2688 inputs so per-element error averages down; and LoRA trains against the quantized base, absorbing systematic shift QLoRA-style. Worth one empirical check: sweep t ∈ [0,1] and compare bf16 vs FP8 output error **on the gate slices specifically** (adaLN-zero gates near 0 are the sensitive ones). If anything the arrow points more aggressive — see D7/E1. |
| H2 | Excluding adaLN from LoRA targets (`lora_minimax_h3.py:31`) | Independently confirmed by Fizgig, with a rationale we could not have derived from the architecture: **ComfyUI's pruned H3 inference builds replace full-width adaLN with a time-embedding curve table, making the tensor `[96768, 8]` instead of `[96768, 2688]`**, so an adaLN LoRA silently drops every key at inference. They measured "~50% likeness until excluded." Record this — it constrains any future decision to train there. |
| H3 | Layer-granular H2D streaming offloader | Substantially ahead of Fizgig's H3 path, which uses plain synchronous `blocks[i].to()` with no pinning, no stream, no prefetch. |

## I. Negative results — measured by others, skip

| Item | Finding |
|---|---|
| `cudnn.benchmark` | A 68× claim that failed to reproduce; on the real model it changed nothing — 66.0 vs 65.9 s planning, 535.1 vs 536.6 ms/step. |
| Shape-bucketed sample ordering | No change at all in eager mode (0.7042 vs 0.7042 s/it). May still matter under `torch.compile`; not enabled without evidence, since grouping correlates consecutive gradients. |
| `enable_gqa=True` in SDPA | Forces the math kernel, ~7× slower than explicitly `repeat_interleave`-ing k/v. (N/A for H3 — no GQA — but relevant to the VAE.) |
| Exact-length attention trimming | Helps compute, destroys shape stability. See F1. |

## J. Audit item

| # | Item |
|---|---|
| J1 | Non-reentrant `torch.utils.checkpoint` **early-stops its recompute** the moment the needed tensors are recovered, so the tail of a checkpointed function never executes during backward. Fizgig found that relying on a tail-placed cleanup "re-accumulated every swapped block on GPU, +9 GB by step 3". Our offloader uses backward hooks and streams rather than in-function `.to()` calls, so the specific bug does not transfer — but audit anything we place *after* the block call inside a checkpointed region. Related: `model.py:410-413` passes only `hidden_states` to the checkpoint and captures `timestep_embedding` in a closure, which is safe **only** because `use_reentrant=False`; fragile if full fine-tuning is ever enabled. |

| J2 | **FIXED (7675fde).** **Audio schedule desynchronizes when the video shift changes.** `training.py:100` calls `map_sigma_between_shifts(video_sigma, source_shift=VIDEO_FLOW_SHIFT, target_shift=AUDIO_FLOW_SHIFT)`, hardcoding 12 as the schedule the incoming sigma is assumed to lie on. That invariant is user-breakable: `--discrete_flow_shift 3` yields video sigma 0.750 with audio 0.4286 where both should be 0.750; and the modes in `_DIRECT_SIGMA_SAMPLING` (`uniform`, `sigmoid`, `logsnr`, the dynamic shifts) never apply `discrete_flow_shift` at all, so the assumption is simply false for them. The remap formula itself is correct -- `shift(unshift(shift(u,12),12),3) == shift(u,3)` exactly -- so this is an unenforced-invariant bug, not a math error. Fix: sample a common unshifted `u` and derive both sigmas from it (`video_sigma = shift(u, cfg_video_shift)`, `audio_sigma = shift(u, AUDIO_FLOW_SHIFT)`), and have `prepare_joint_noisy_inputs` accept both explicit sigmas rather than inferring provenance from a global constant. This is a correctness bug and blocks F5/F6. |

| J3 | The dynamic-shift `--timestep_sampling` modes (`flux_shift`, `qwen_shift`, `krea2_shift`, `ideogram4_shift`, `qinglong_*`) compute `shift = exp(mu)` themselves, independently of `--discrete_flow_shift`, so they hand H3 an already-shifted coordinate and H3's per-modality shift is then applied on top. Both modalities stay synchronized (no correctness bug), but the resulting marginal is doubly shifted. They also derive `mu` from only the two trailing latent dims, ignoring H3's temporal extent, so they are meaningless for H3 regardless. Documented as unsupported in `minimax_h3.md`; consider rejecting them outright alongside F6. |

## E5. AdaLN low-rank factorization — SHIPPED (`1e8ec0a`, `--h3_adaln_rank`)

`adaln_proj` input is `a(t) = silu(time_embedder(t))`, a smooth 1-D curve traced by the scalar
timestep (H3 passes `t = 1 - sigma` in `[0,1]`, sinusoidal frequencies 1.0 down to 1e-4). It is
therefore low-rank, and the projection can be stored as `[96768, r]` instead of `[96768, 2688]`.

    a(t) ~= mean + U c(t)
    W a(t) + b ~= (W U) c(t) + (b + W mean)      <- the W mean term MUST be folded into the bias

**Validated against the released BF16 checkpoint, all 50 blocks**, error on the modulation output
`W a(t) + b`, measured per block and per (modality, modulation-parameter) group:

| | mean | max | worst single group |
|---|---|---|---|
| rank 8 | 1.271e-05 | 1.533e-05 | 6.81e-05 (block 1, audio, shift_mlp) |
| rank 16 | 5.916e-07 | 6.260e-07 | 8.05e-07 (block 22, audio, gate_mlp) |
| rank 32 | identical to rank 16 (float32 measurement floor) | | |
| **FP8 e4m3 block-64 — already shipped on these weights** | **4.945e-03** | **6.390e-03** | |

Rank 8 is **389x below** the FP8 error already accepted; rank 16 is **8358x below**. Error is uniform
across depth and actually declines with block index. Curve-space error at rank 8 is 4.5e-05 but
output-space error is 1.3e-05, i.e. `W` **attenuates** the residual ~3.5x rather than amplifying it
(this was the mandatory open question from review).

Parameter effect: 13.01 B -> 38.7 M (rank 8) or 77.4 M (rank 16). Whole model 33.12 B -> ~20.1 B.
BF16 66.2 GB -> ~40.2 GB; FP8 33.1 GB -> ~20.1 GB. Under layer streaming the steady-state VRAM win
is only one or two adaLN payloads, but it removes ~13 GB of H2D traffic per forward and shrinks the
ring slots ~1.7x (adaLN currently sizes them).

Resolution as shipped: the two candidate designs (basis factorization vs the ComfyUI
`adaln_t_table` form) were merged rather than chosen between. `adaln_lowrank.py` fits the basis and
contracts each weight; the runtime consumes the result through the existing `adaln_t_table` path, so
the reduced model is checkpoint-compatible with ComfyUI's pruned form while being derived from the
released BF16 weights at load. No converter and no second download.

The factorization runs as a `WeightTransformHooks.split_hook` during the streaming load, before any
quantization, so it sees full checkpoint precision. The basis is fitted **uncentered**
(`center=False`), which makes `mean` zero — the `W mean` term then vanishes and biases pass through
untouched, removing the hook's dependence on weights and biases arriving in a particular order.

Former open items, all closed: the reduced adaln is excluded from FP8 targets (`model_loader.py:244`);
it remains an `nn.Linear` so the offloader's class-name filter still finds it; `W @ U` is formed in
float64 over row chunks before any cast; `--h3_adaln_rank` refuses checkpoints that are already
pruned or INT8 ConvRot rather than reducing twice.

## K. Measured fidelity of the memory-saving options

Identical fixed-seed inputs through all 50 blocks of the released FL2VA BF16 transformer; relative L2
of the predictions against the BF16 reference. Also reproduced in `docs/minimax_h3.md`.

| configuration | video | audio | peak VRAM |
|---|---|---|---|
| BF16 | reference | | 61.73 GiB |
| `--h3_adaln_rank 16` | **4.30e-02** | **5.71e-02** | 37.72 GiB |
| INT8 ConvRot checkpoint | 8.08e-02 | 1.41e-01 | 31.73 GiB |
| `--fp8_base` | 1.25e-01 | 2.39e-01 | 33.35 GiB |

Every option changes the output at the percent level: 50 residual blocks amplify any small weight
perturbation. That is normal, not a defect — establishing it cost most of a session, because the
reduction was first judged against bf16's rounding step with no baseline to compare against.

**Not measured by any of this:** whether an adapter trained against one base transfers to a
differently quantized or reduced base. Applies to every row; needs matched training runs.

## K2. OPEN DEFECT — reduced adaLN is stored at bf16, which dominates its error

Measured against the released pruned checkpoint (`rockerBOO/minimax-h3-nvfp4`,
`minimax_h3_fl2va_pruned_nvfp4.safetensors`). Its adaLN is **not** NVFP4: `adaln_t_table` is F32
`[1025, 8]` and `adaln_proj.linear.weight` is **F16** `[96768, 8]`, so its pruning error is isolable
from its quantization error. `final_layer.adaln_proj` is pruned there too — as it is here, since our
hook matches by key suffix. This closes that E5 open item.

Relative error of the modulation `W a(t) + b` vs the float64 BF16 reference, dense 977-point grid
deliberately off the table nodes, blocks 0/10/20/30/40:

| storage of the reduced weight | mean rel error |
|---|---|
| **bf16 — currently shipped** | **1.083e-03** |
| theirs (rank 8, fp16) | 2.029e-04 |
| fp16, our rank 16 | 1.354e-04 |
| fp32, our rank 16 | 4.235e-08 |

The basis is not the limit — at fp32 it is essentially exact. `storage_dtype=torch.bfloat16` in
`make_adaln_split_hook` is: bf16's 8-bit mantissa gives ~1.1e-3 RMS, exactly the observed floor.
Two symptoms follow. Our rank 8 and rank 16 score identically, so the extra rank currently buys
nothing. And **as shipped we are 5.3x less precise than the published pruned checkpoint.**

Fix: store fp32 and compute the modulation in fp32, casting the result to bf16.

fp16 is **not** a cheaper middle option, despite scoring 1.354e-04. Activations are bf16 and
`F.linear` requires matching dtypes, so fp16 needs exactly the same compute-path change as fp32
while giving up ~3200x the accuracy to save 155 MB on a ~20 GB model. (`max |W_reduced|` is 49.4,
so fp16 would be safe on range — limit 65504 — it is simply pointless.) Nor can the defect be fixed
by keeping bf16 storage and raising only the matmul precision: the error is committed when `W @ U`
is rounded to bf16, and no downstream precision recovers it. The stored dtype is the one that
matters.

The modulation input is ~6 rows wide, so the fp32 matmul is arithmetically free, and 310 MB is
negligible against the 13.0 GB -> 0.3 GB the reduction already saved.

FIXED. `storage_dtype` now defaults to float32 and the weight branch honours it. Verified end to
end on all 50 blocks against the released BF16 checkpoint:

| variant | video relL2 | audio relL2 | peak |
|---|---|---|---|
| rank16 float32 | 4.3029e-02 | 5.7079e-02 | 37.72 GiB |
| rank16 bfloat16 | 5.0782e-02 | 6.9357e-02 | 37.57 GiB |

The float32 row reproduces section K's 4.30e-02 exactly, so K was measured before the bf16 cast was
introduced: this was a regression against a documented number, not a stale measurement.

Why the end-to-end gain is ~18% while the modulation error improves by ~25000x: the BF16 reference
computes its own modulation in bf16, so it carries the same ~1e-3 error the fix removes. The
forward-pass delta against BF16 is therefore floored by the reference's own rounding, and cannot
show the full improvement. Measured against a float64 modulation the reduced path is now closer to
exact than the BF16 checkpoint itself.

Harness: `/mnt/data2/sgornostaev/h3-adaln/pruned_ref/` (`fetch.py` pulls only the adaLN tensors by
HTTP range request, ~83 MB; `compare.py`, `dtype_test.py`).

## L. Next candidate: online INT8 ConvRot, composed with E5

The published INT8 checkpoints quantize AdaLN at **full rank** — verified: the files on the workbox
have no `adaln_t_table`, keep `time_embedder`, and carry `blocks.N.adaln_proj.linear.weight` as
`[96768, 2688]` I8. So much of their 8.08e-02 comes from pushing 13.0B AdaLN parameters through INT8,
which is exactly what E5 removes.

Reducing AdaLN first (BF16, 0.15 GB) and applying ConvRot only to the remaining ~20B should be both
smaller and more faithful than the published checkpoint — estimated ~19 GiB. Arithmetic, not measured.

Blockers: `int8_convrot=True` reads pre-quantized `.comfy_quant`/`.weight_scale` tensors from the
file, so no path quantizes BF16 to ConvRot at load; the loader rejects `int8_convrot` with
`fp8_scaled`; E5 rejects `int8_convrot` outright on the assumption those checkpoints ship pre-reduced
(they do not). ConvRot rotates the input side — the same side E5 factorizes — so the basis would need
fitting in the rotated space. The low-rank structure should survive an orthogonal rotation, but that
is reasoning, not measurement.

## M. Reproduction environment

- Workbox task dir: `/mnt/data2/sgornostaev/h3-adaln/h3-tuner`, with `USES.txt`. Self-contained and
  deletable. Do **not** use `/mnt/data1/.../h3-experiments/source/h3-tuner` — different lineage.
- Python: `/mnt/data2/sgornostaev/comfy/.venv/bin/python`. The `minimax-h3` venv lacks `accelerate`.
- BF16 checkpoints: `/mnt/data1/sgornostaev/h3-experiments/models/Comfy-MiniMax-H3/`.
- Shared box — check free VRAM first. The BF16 reference needs ~62 GiB.
- Measurement harness: `G:\repos\h3-adaln-harness\` — `fwd_compare.py` (produced every number in
  section K), `validate_adaln.py` (per-block modulation error), and a `README.md` with the exact
  invocation. Deliberately outside the repo so it is neither committed nor removed by `git clean`.

---

## Suggested sequencing

Revised after the workload calibration above. The previous ordering front-loaded sync and
memory-traffic fixes on the assumption they were speed wins; at H3's token counts they are not.

**Phase 1 — speed, where the FLOPs actually are.** B1 (flash / sage attention: attention is ~52%
of forward FLOPs at `S ≈ 29k`, ~21% at `S ≈ 9k`), then B5/D4 (FP8 `_scaled_mm` — today our FP8
dequantizes to bf16 and computes at bf16 speed, so this is pure upside) with D5 and D8 alongside.

**Phase 2 — VRAM headroom, so larger clips fit at all.** B2 (`--gradient_checkpointing_cpu_offload`,
worth ~15 GB of block-boundary activations at `S ≈ 29k`), then A4, A1, A17, A3, A5 as a batch —
each is a modest VRAM win and together they cut the intra-block backward peak substantially.
E1/D7 if adaLN residency becomes the binding constraint.

**Phase 3 — cheap hygiene, batched.** C1, C2, and the sync/allocation items A6–A16. Individually
sub-0.1%; worth doing as one pass because A6 and F2 also remove `torch.compile` graph breaks, and
C1 protects against fragmentation degradation on long runs.

**Phase 4 — compile.** C3 + C5 + C6 as prerequisites (without C3 dynamo silently runs eager), then
B3, then C4. F1/F2 make it amortize. Budget single-digit percent, not Fizgig's 2×.

**Phase 5 — architectural.** G1 (TREAD token dropping) is the largest remaining speed lever by a
wide margin. Then D1 (INT8 W8A8) if FP8 `_scaled_mm` proves insufficient, E2/E3 (fused kernels),
G3 (planner).

**Independent of all phases:** F5 (`discrete_flow_shift`) — zero performance cost, potentially large
convergence effect, and the finding with the strongest independent verification in this document.
Should land before any long training run, not after.

