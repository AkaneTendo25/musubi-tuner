"""CPU checks of the ``--h3_profile_steps`` kernel table and its flag."""

from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.model import h3_profile_scope
from musubi_tuner.minimax_h3_train_network import (
    H3_PROFILE_WAIT_STEPS,
    H3_PROFILE_WARMUP_STEPS,
    H3StepProfiler,
    MiniMaxH3NetworkTrainer,
    attribute_h3_profile_scopes,
    categorize_h3_profile_rows,
    collect_h3_profile_rows,
    create_parser,
    device_busy_us,
    format_h3_profile_scopes,
    format_h3_profile_table,
    format_h3_profile_top_kernels,
)


def test_profile_rows_are_sorted_into_categories_by_kernel_name():
    rows = [
        ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", 10.0),
        ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x3_tn", 5.0),
        ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm>", 2.0),
        ("aten::matmul", 1.0),
        ("flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4>>", 20.0),
        ("fmha_cutlassF_bf16_aligned_64x128_rf_sm80", 7.0),  # cutlass name, but attention
        ("cudnn_generated_fort_native_sdpa_sm90_flash_fprop", 3.0),
        ("Memcpy HtoD (Pinned -> Device)", 30.0),
        ("Memcpy DtoH (Device -> Pageable)", 4.0),
        ("Memcpy DtoD (Device -> Device)", 6.0),  # neither swap direction
        ("void at::native::vectorized_elementwise_kernel<4, ...>", 8.0),
        ("void at::native::reduce_kernel<512, 1, ...>", 1.5),
    ]

    categories = categorize_h3_profile_rows(rows)

    assert list(categories) == ["attention", "gemm", "swap", "elementwise/other"]
    assert categories["gemm"] == (18.0, 4)
    assert categories["attention"] == (30.0, 3)
    assert categories["swap"] == (34.0, 2)
    assert categories["elementwise/other"] == (15.5, 3)


def test_profile_rows_recognise_hopper_cublas_and_flash_attention_kernel_names():
    rows = [
        ("nvjet_tst_128x256_64x4_2x1_v_bz_TNT", 100.0),
        (
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
            50.0,
        ),
        ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>", 20.0),
        ("gemmSN_TN_kernel_64x64_16x4_NN_vec", 10.0),
        ("void gemv2T_kernel_val<int, int, __nv_bfloat16, ...>", 5.0),
        ("Cijk_Alik_Bljk_BBS_BH_MT64x64x16_MI16x16x16x1_SN_1LDSB0", 2.0),
        ("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...>, true, false, true>", 300.0),
        ("void flash::compute_attn_ws<flash::CollectiveMainloopFwdSm90<...>>", 100.0),
        ("flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<...>>", 200.0),
        ("void pytorch_flash::flash_fwd_splitkv_kernel<...>", 30.0),
        ("fmha_cutlassF_bf16_aligned_64x128_rf_sm80", 7.0),
        ("cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_kernel0_0", 3.0),
        ("_int8_attention_fwd_kernel", 4.0),
        ("_int8_attention_dkdv_kernel", 4.0),
        ("Memcpy HtoD (Pinned -> Device)", 60.0),
        ("Memcpy DtoH (Device -> Pinned)", 6.0),
        ("Memcpy DtoD (Device -> Device)", 9.0),
        ("void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<...>>", 40.0),
        ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<...>>", 20.0),
        ("_swiglu_fwd", 15.0),
        ("_indexed_adaln_rmsnorm_bwd", 8.0),
        ("triton_poi_fused_add_mul_0", 1.0),
    ]

    categories = categorize_h3_profile_rows(rows)

    assert categories["gemm"] == (187.0, 6)
    assert categories["attention"] == (648.0, 8)
    assert categories["swap"] == (66.0, 2)
    assert categories["elementwise/other"] == (93.0, 6)


def _event(name, start, end, *, device="cuda", annotation=False, async_=False, self_cpu=0.0):
    device_type = torch.autograd.DeviceType.CUDA if device == "cuda" else torch.autograd.DeviceType.CPU
    return SimpleNamespace(
        name=name,
        device_type=device_type,
        is_user_annotation=annotation,
        is_async=async_,
        time_range=SimpleNamespace(start=start, end=end),
        self_cpu_time_total=self_cpu,
    )


def test_collected_rows_are_leaf_kernels_only_and_partition_the_kernel_time():
    """Labels and step markers are user annotations spanning their kernels, and CPU ops carry the
    device time of the kernels they launch: none of them may count, or the table sums past 100%."""
    events = [
        _event("ProfilerStep#3", 0.0, 12.0, annotation=True),
        _event("h3.attn", 0.0, 9.0, annotation=True),
        _event("h3.lora", 0.0, 5.0, annotation=True),
        _event("aten::mm", 0.0, 5.0, device="cpu", self_cpu=1.0),
        _event("nvjet_tst_128x256_64x4_2x1_v_bz_TNT", 0.0, 5.0),
        _event("void flash::flash_fwd_kernel<...>", 5.0, 9.0),
        _event("Memcpy HtoD (Pinned -> Device)", 2.0, 8.0),  # overlaps both on the copy stream
        _event("nvjet_tst_128x256_64x4_2x1_v_bz_TNT", 10.0, 11.0),
        _event("void at::native::vectorized_elementwise_kernel<...>", 11.0, 11.5),
        _event("cudaLaunchKernel", 0.0, 0.1, async_=True),
    ]

    rows, busy_us = collect_h3_profile_rows(events, "cuda")

    assert [(name, count) for name, _, count in rows] == [
        ("nvjet_tst_128x256_64x4_2x1_v_bz_TNT", 2),
        ("Memcpy HtoD (Pinned -> Device)", 1),
        ("void flash::flash_fwd_kernel<...>", 1),
        ("void at::native::vectorized_elementwise_kernel<...>", 1),
    ]
    categories = categorize_h3_profile_rows(rows)
    kernel_sum = sum(time_us for _, time_us, _ in rows)
    assert kernel_sum == 16.5
    assert sum(time_us for time_us, _ in categories.values()) == kernel_sum
    assert categories["gemm"] == (6.0, 1)
    assert categories["attention"] == (4.0, 1)
    assert categories["swap"] == (6.0, 1)
    assert categories["elementwise/other"] == (0.5, 1)
    assert busy_us == 10.5  # [0, 9] and [10, 11.5]; the annotations do not stretch it to 12


def _cpu_op(name, start, end, *, id, thread=1, sequence_nr=-1, scope=0, fwd_thread=0, parent=None, self_cpu=0.0):
    event = _event(name, start, end, device="cpu", self_cpu=self_cpu)
    event.id = id
    event.thread = thread
    event.sequence_nr = sequence_nr
    event.scope = scope
    event.fwd_thread = fwd_thread
    event.cpu_parent = parent
    event.linked_correlation_id = 0
    return event


def _kernel(name, start, end, *, id=None, linked=None):
    """A device kernel event. torch 2.9 gives it only its CUPTI correlation ``id`` (shared with the
    ``cudaLaunchKernel`` runtime record); torch >= 2.10 adds ``linked_correlation_id`` naming the op."""
    event = _event(name, start, end)
    if id is not None:
        event.id = id
    if linked is not None:
        event.linked_correlation_id = linked
    return event


def _label(name, start, end, *, device="cuda", thread=1):
    event = _event(name, start, end, device=device, annotation=True)
    event.thread = thread
    return event


# The shapes a kernel-to-op link takes across torch releases, and a trace without any link. ``op``
# is the launching op's id; ``launch`` a runtime-record id the kernel shares under torch 2.9.
_LINK_SHAPES = {
    "unlinked": lambda op, launch: {},
    "runtime_id": lambda op, launch: {"id": launch},
    "linked_correlation_id": lambda op, launch: {"linked": op},
}


def _step_trace(shape):
    """One profiled step: a student forward, then a checkpoint recompute, the backward, the optimizer."""
    link = _LINK_SHAPES[shape]
    # CPU side: the forward thread's labels and ops (sequence numbers pair forward ops with backward nodes).
    lora_mm = _cpu_op("aten::mm", 23.0, 25.0, id=101, sequence_nr=7)
    gate_add = _cpu_op("aten::add", 50.0, 52.0, id=102, sequence_nr=8)
    attn_op = _cpu_op("aten::_scaled_dot_product_flash_attention", 45.0, 47.0, id=103, sequence_nr=9)
    # Backward thread: a checkpoint recompute (labels re-entered) and the LoRA matmul's own backward node.
    recompute_node = _cpu_op("CheckpointFunctionBackward", 480.0, 530.0, id=200, thread=2, scope=1, sequence_nr=8, fwd_thread=1)
    recompute_attn = _cpu_op("aten::_scaled_dot_product_flash_attention", 500.0, 502.0, id=201, thread=2, parent=recompute_node)
    mm_backward = _cpu_op("MmBackward0", 590.0, 620.0, id=210, thread=2, scope=1, sequence_nr=7, fwd_thread=1)
    mm_backward_op = _cpu_op("aten::mm", 600.0, 602.0, id=202, thread=2, parent=mm_backward)
    orphan_node = _cpu_op("SomeBackward0", 640.0, 660.0, id=211, thread=2, scope=1, sequence_nr=99, fwd_thread=1)
    orphan_op = _cpu_op("aten::mul", 645.0, 646.0, id=203, thread=2, parent=orphan_node)
    optimizer_op = _cpu_op("aten::_fused_adamw_", 700.0, 702.0, id=301)
    # The runtime records the torch 2.9 shape links through: same correlation id as the kernel, nested under the op.
    runtime = [
        _cpu_op("cudaLaunchKernel", op.time_range.start + 0.5, op.time_range.start + 0.6, id=1000 + op.id, parent=op)
        for op in (lora_mm, attn_op, gate_add, recompute_attn, mm_backward_op, orphan_op, optimizer_op)
    ]
    return [
        _event("ProfilerStep#3", 0.0, 800.0, annotation=True),
        _label("h3.forward.student", 0.0, 100.0, device="cpu"),
        _label("h3.block", 10.0, 60.0, device="cpu"),
        _label("h3.attn", 20.0, 48.0, device="cpu"),
        _label("h3.lora", 22.0, 30.0, device="cpu"),
        lora_mm,
        attn_op,
        gate_add,
        recompute_node,
        recompute_attn,
        mm_backward,
        mm_backward_op,
        orphan_node,
        orphan_op,
        optimizer_op,
        *runtime,
        # Device side: the labels as user annotations on the device timeline, and the kernels.
        _label("h3.forward.student", 200.0, 400.0),
        _label("h3.block", 210.0, 300.0),
        _label("h3.attn", 220.0, 260.0),
        _label("h3.lora", 222.0, 240.0),
        _kernel("Memcpy HtoD (Pinned -> Device)", 203.0, 206.0),
        _kernel("nvjet_tst_128x256", 225.0, 235.0, **link(101, 1101)),
        _kernel("void flash::flash_fwd_kernel<...>", 245.0, 255.0, **link(103, 1103)),
        _kernel("void at::native::vectorized_elementwise_kernel<add>", 270.0, 280.0, **link(102, 1102)),
        _label("h3.attn", 495.0, 520.0),  # re-entered by the checkpoint recompute, no role around it
        _kernel("void flash::flash_fwd_kernel<...>", 505.0, 515.0, **link(201, 1201)),
        _kernel("nvjet_tst_128x256", 605.0, 615.0, **link(202, 1202)),
        _kernel("void at::native::vectorized_elementwise_kernel<mul>", 650.0, 652.0, **link(203, 1203)),
        _label("Optimizer.step#AdamW.step", 700.0, 720.0),
        _kernel("multi_tensor_apply_kernel", 705.0, 715.0, **link(301, 1301)),
        _event("cudaLaunchKernel", 0.0, 0.1, async_=True),
    ]


_FORWARD_ROWS = {
    ("h3.lora", "student"): (10.0, 1),
    ("h3.attn", "student"): (10.0, 1),
    ("h3.block", "student"): (10.0, 1),
    ("(no h3 scope)", "student"): (3.0, 1),
    ("h3.attn", "recompute"): (10.0, 1),
    ("(no h3 scope)", "other"): (10.0, 1),
}


def test_kernels_are_attributed_to_the_innermost_scope_per_phase_without_a_launch_link():
    """torch 2.9's kernel events carry no link to the launching op: forward kernels still belong to the
    innermost ``h3.*`` label enclosing them and to the ``h3.forward.<role>`` label as their phase, a label
    re-entered inside the backward window is a recompute, and the window's unlabelled kernels are reported
    as one unattributed backward row rather than guessed."""
    scopes = attribute_h3_profile_scopes(_step_trace("unlinked"), "cuda")

    assert scopes["linked"] is False
    assert scopes["forwards"] == {"student": 1}
    assert scopes["labels"] == {
        ("h3.block", "student"): 1,
        ("h3.attn", "student"): 1,
        ("h3.lora", "student"): 1,
        ("h3.attn", "recompute"): 1,
    }
    rows = {key: (time_us, launches) for key, (time_us, launches, _) in scopes["rows"].items()}
    assert rows == {**_FORWARD_ROWS, ("(no h3 scope)", "backward (unattributed)"): (12.0, 2)}
    assert scopes["rows"][("h3.lora", "student")][2] == {"gemm": 10.0}
    assert scopes["rows"][("h3.attn", "student")][2] == {"attention": 10.0}
    assert scopes["rows"][("(no h3 scope)", "student")][2] == {"swap": 3.0}
    assert sum(time_us for time_us, _ in rows.values()) == 65.0  # every kernel counted exactly once

    text = format_h3_profile_scopes(scopes, wall_us=1000.0, steps=2)
    lines = text.splitlines()
    assert "per-scope backward attribution is UNAVAILABLE on this torch" in lines[0]
    assert "recompute detection is time-window based" in lines[0]
    assert lines[1] == "forwards in the window: student 1"
    assert lines[2].split() == [
        "scope", "phase", "total", "ms", "ms/step", "share", "launches", "labels", "gemm", "ms", "attn", "ms", "elem", "ms", "swap", "ms"
    ]  # fmt: skip
    lora = next(line for line in lines if line.startswith("h3.lora          student"))
    assert lora.split() == ["h3.lora", "student", "0.0", "0.0", "1.0%", "1", "1", "0.0", "0.0", "0.0", "0.0"]
    unattributed = next(line for line in lines[3:] if "backward (unattributed)" in line)
    assert unattributed.split()[:6] == ["(no", "h3", "scope)", "backward", "(unattributed)", "0.0"]


@pytest.mark.parametrize("shape", ["runtime_id", "linked_correlation_id"])
def test_backward_kernels_are_charged_to_their_forward_scope_when_the_trace_links_them(shape):
    """With a launch link -- ``linked_correlation_id`` on torch >= 2.10, the shared correlation id of the
    ``cudaLaunchKernel`` record on 2.9 -- a bare backward kernel is charged to the label of the forward op
    its autograd node differentiates, and a node without a forward match stays unscoped but is backward."""
    scopes = attribute_h3_profile_scopes(_step_trace(shape), "cuda")

    assert scopes["linked"] is True
    rows = {key: (time_us, launches) for key, (time_us, launches, _) in scopes["rows"].items()}
    assert rows == {**_FORWARD_ROWS, ("h3.lora", "backward"): (10.0, 1), ("(no h3 scope)", "backward"): (2.0, 1)}

    lines = format_h3_profile_scopes(scopes, wall_us=1000.0, steps=2).splitlines()
    assert "UNAVAILABLE" not in lines[0]
    backward = next(line for line in lines if line.startswith("h3.lora          backward"))
    assert backward.split()[5:7] == ["1", "0"]


def test_recompute_is_only_a_label_re_entered_inside_the_backward_window():
    """A label outside every role is a recompute only after the step's last forward and before its
    optimizer label; elsewhere it is forward work that simply carries no role."""
    events = [
        _event("ProfilerStep#1", 0.0, 100.0, annotation=True),
        _label("h3.attn", 5.0, 15.0),  # before any role: unroled forward work (e.g. a validation pass)
        _kernel("k_early", 6.0, 8.0),
        _label("h3.forward.student", 20.0, 40.0),
        _label("h3.attn", 22.0, 30.0),
        _kernel("k_fwd", 24.0, 26.0),
        _label("h3.attn", 50.0, 60.0),  # after the forward, before the optimizer: recompute
        _kernel("k_recompute", 52.0, 54.0),
        _kernel("k_bwd", 65.0, 70.0),
        _label("Optimizer.step#AdamW.step", 80.0, 90.0),
        _kernel("k_optimizer", 82.0, 84.0),
        _event("ProfilerStep#2", 100.0, 200.0, annotation=True),
        _label("h3.attn", 105.0, 115.0),  # a step with no forward role has no backward window
        _kernel("k_next_step", 106.0, 108.0),
    ]

    scopes = attribute_h3_profile_scopes(events, "cuda")

    rows = {key: launches for key, (_, launches, _) in scopes["rows"].items()}
    assert rows == {
        ("h3.attn", "forward (unroled)"): 2,
        ("h3.attn", "student"): 1,
        ("h3.attn", "recompute"): 1,
        ("(no h3 scope)", "backward (unattributed)"): 1,
        ("(no h3 scope)", "other"): 1,
    }
    assert scopes["labels"] == {("h3.attn", "forward (unroled)"): 2, ("h3.attn", "student"): 1, ("h3.attn", "recompute"): 1}


def test_scope_attribution_ignores_labels_that_only_partially_overlap_a_kernel():
    events = [
        _label("h3.attn", 0.0, 10.0),
        _label("h3.mlp", 8.0, 20.0),  # overlaps h3.attn without nesting in it
        _kernel("k_attn", 2.0, 4.0),
        _kernel("k_mlp", 12.0, 14.0),
        _kernel("k_straddle", 9.0, 11.0),  # inside h3.mlp, only starts inside h3.attn
        _kernel("k_after", 30.0, 31.0),
    ]

    rows = {key: launches for key, (_, launches, _) in attribute_h3_profile_scopes(events, "cuda")["rows"].items()}

    assert rows == {("h3.attn", "forward (unroled)"): 1, ("h3.mlp", "forward (unroled)"): 2, ("(no h3 scope)", "other"): 1}


def test_scope_attribution_uses_cpu_ops_and_labels_without_a_device():
    outer = _cpu_op("aten::linear", 1.0, 9.0, id=1, self_cpu=2.0)
    inner = _cpu_op("aten::addmm", 2.0, 8.0, id=2, self_cpu=6.0, parent=outer)
    events = [_label("h3.forward.student", 0.0, 20.0, device="cpu"), _label("h3.lora", 0.0, 10.0, device="cpu"), outer, inner]

    scopes = attribute_h3_profile_scopes(events, "cpu")

    assert scopes["rows"] == {("h3.lora", "student"): (8.0, 2, {"elementwise/other": 8.0})}
    assert scopes["labels"] == {("h3.lora", "student"): 1}


def test_top_kernels_are_listed_by_device_time_with_their_category():
    rows = [("small_kernel", 1.0, 3), ("nvjet_tst_big", 5000.0, 2), ("flash_fwd_kernel", 3000.0, 1)]

    text = format_h3_profile_top_kernels(rows, limit=2)

    lines = text.splitlines()
    assert lines[0] == "top 2 kernels by device time (of 3 distinct)"
    assert lines[2].split() == ["5.0", "2", "gemm", "nvjet_tst_big"]
    assert lines[3].split() == ["3.0", "1", "attention", "flash_fwd_kernel"]
    assert len(lines) == 4


def test_device_busy_is_the_union_of_overlapping_kernel_intervals():
    # a swap copy (0..8) overlapping a GEMM (2..5) and an attention kernel (5..9), then a gap
    intervals = [(2.0, 5.0), (0.0, 8.0), (5.0, 9.0), (12.0, 13.0)]

    assert device_busy_us(intervals) == 10.0
    assert device_busy_us([]) == 0.0
    assert device_busy_us([(3, 4)]) == 1.0


def test_profile_table_reports_idle_as_the_wall_time_outside_the_kernel_union():
    # 6 ms of GEMM fully overlapped by an 8 ms swap copy on another stream: durations sum to 14 ms,
    # the device was busy for 8 ms, and idle is the remaining 2 ms of the window.
    categories = categorize_h3_profile_rows([("gemm_a", 6000.0), ("Memcpy HtoD (Pinned -> Device)", 8000.0)])

    table = format_h3_profile_table(categories, wall_us=10000.0, steps=2, busy_us=8000.0)

    lines = table.splitlines()
    assert lines[0].startswith("profile of 2 optimizer step(s), wall 10.0 ms (5.0 ms/step), device busy 8.0 ms")
    assert "kernel durations sum to 14.0 ms" in lines[0]
    idle = [line for line in lines if line.startswith("idle")]
    assert len(idle) == 1
    assert idle[0].split() == ["idle", "2.0", "1.0", "20.0%", "0"]
    gemm = next(line for line in lines if line.startswith("gemm"))
    assert gemm.split() == ["gemm", "6.0", "3.0", "60.0%", "1"]
    swap = next(line for line in lines if line.startswith("swap"))
    assert swap.split() == ["swap", "8.0", "4.0", "80.0%", "1"]


def test_profile_table_never_reports_negative_idle():
    table = format_h3_profile_table(categorize_h3_profile_rows([("gemm", 500.0)]), wall_us=100.0, steps=1, busy_us=500.0)

    assert next(line for line in table.splitlines() if line.startswith("idle")).split()[1] == "0.0"


def test_profile_steps_flag_rejects_a_run_shorter_than_the_window():
    trainer = MiniMaxH3NetworkTrainer()
    needed = H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS + 4

    args = create_parser().parse_args(["--sdpa", "--h3_profile_steps", "4", "--max_train_steps", str(needed - 1)])
    with pytest.raises(ValueError, match="raise --max_train_steps or lower --h3_profile_steps"):
        trainer.handle_model_specific_args(args)

    args = create_parser().parse_args(["--sdpa", "--h3_profile_steps", "4", "--max_train_steps", str(needed)])
    trainer.handle_model_specific_args(args)

    # With an epoch count the step total is only known at train start, where it is checked again.
    args = create_parser().parse_args(["--sdpa", "--h3_profile_steps", "4", "--max_train_steps", "1", "--max_train_epochs", "1"])
    trainer.handle_model_specific_args(args)
    args.max_train_steps = needed - 1
    with pytest.raises(ValueError, match="raise --max_train_steps or lower --h3_profile_steps"):
        trainer.on_train_start(args, SimpleNamespace(is_main_process=True, device=torch.device("cpu")), None, None, None)
    assert trainer._h3_profiler is None


def test_profile_steps_flag_rejects_negative_values():
    args = create_parser().parse_args(["--sdpa", "--h3_profile_steps", "-1"])

    with pytest.raises(ValueError, match="--h3_profile_steps must be non-negative"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_profile_steps_flag_defaults_to_off_and_installs_no_profiler(tmp_path):
    args = create_parser().parse_args(["--sdpa", "--output_dir", str(tmp_path), "--output_name", "run"])
    assert args.h3_profile_steps == 0
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cpu"))

    trainer.on_train_start(args, accelerator, None, None, None)
    trainer.on_post_optimizer_step(args, accelerator, None, None, True, 0)

    assert trainer._h3_profiler is None
    assert not list(tmp_path.iterdir())


def test_profile_scope_is_a_no_op_without_a_profiler():
    with h3_profile_scope("h3.attn") as scope:
        pass
    assert scope is None


def test_profiler_writes_one_table_after_the_active_window_and_stops(tmp_path, caplog):
    args = create_parser().parse_args(
        ["--sdpa", "--output_dir", str(tmp_path / "out"), "--output_name", "run", "--h3_profile_steps", "2"]
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cpu"))
    trainer.on_train_start(args, accelerator, None, None, None)
    assert isinstance(trainer._h3_profiler, H3StepProfiler)
    profiler = trainer._h3_profiler
    weight = torch.randn(32, 32)

    total = H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS + 2
    with caplog.at_level("INFO"):
        for step in range(total):
            with h3_profile_scope("h3.attn"):
                torch.randn(32, 32) @ weight
            # an accumulation micro-step does not advance the profiler
            trainer.on_post_optimizer_step(args, accelerator, None, None, False, step)
            assert trainer._h3_profiler is profiler
            trainer.on_post_optimizer_step(args, accelerator, None, None, True, step)
            if step < total - 1:
                assert trainer._h3_profiler is profiler and not profiler.finished

    assert profiler.finished
    assert trainer._h3_profiler is None
    written = tmp_path / "out" / "run_profile.txt"
    text = written.read_text(encoding="utf-8")
    assert text.startswith(profiler.table + "\n\n")
    assert "kernels by device time" in text
    assert "aten::mm" in text  # the CPU stand-in rows list the ops themselves
    scope_section, kernel_section = text.split("kernels by device time")
    assert "device time by innermost h3.* scope" in scope_section
    assert "h3.attn" in scope_section  # the label heads a scope row ...
    assert "h3.attn" not in kernel_section  # ... and is an annotation, never a kernel row
    lines = profiler.table.splitlines()
    assert lines[0].startswith("profile of 2 optimizer step(s), wall ")
    assert [line.split()[0] for line in lines[2:]] == ["attention", "gemm", "swap", "elementwise/other", "idle"]
    assert "--h3_profile_steps table" in caplog.text

    # Training continues untouched once the table is out, and train-end finalisation has nothing to do.
    trainer.on_post_optimizer_step(args, accelerator, None, None, True, total)
    assert trainer._h3_profiler is None
    trainer._finalize_h3_profiler()
    assert written.read_text(encoding="utf-8") == text


def _profiling_trainer(tmp_path, steps: int):
    args = create_parser().parse_args(
        ["--sdpa", "--output_dir", str(tmp_path / "out"), "--output_name", "run", "--h3_profile_steps", str(steps)]
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    accelerator = SimpleNamespace(is_main_process=True, device=torch.device("cpu"))
    trainer.on_train_start(args, accelerator, None, None, None)
    return args, trainer, accelerator


def test_train_end_writes_the_partial_window_when_training_stops_inside_it(tmp_path, caplog):
    args, trainer, accelerator = _profiling_trainer(tmp_path, steps=3)
    profiler = trainer._h3_profiler
    weight = torch.randn(32, 32)

    # two wait, one warmup, then ONE of the three active steps before training ends
    with caplog.at_level("INFO"):
        for step in range(H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS + 1):
            torch.randn(32, 32) @ weight
            trainer.on_post_optimizer_step(args, accelerator, None, None, True, step)
        assert not profiler.finished
        trainer._finalize_h3_profiler()

    assert profiler.finished
    assert trainer._h3_profiler is None
    text = (tmp_path / "out" / "run_profile.txt").read_text(encoding="utf-8")
    assert text.startswith("profile of 1 optimizer step(s) of the 3 requested, INCOMPLETE, wall ")
    assert "idle" in text
    assert "INCOMPLETE" in caplog.text


def test_train_end_reports_a_window_that_never_opened(tmp_path, caplog):
    args, trainer, accelerator = _profiling_trainer(tmp_path, steps=2)
    profiler = trainer._h3_profiler

    with caplog.at_level("WARNING"):
        trainer.on_post_optimizer_step(args, accelerator, None, None, True, 0)
        trainer._finalize_h3_profiler()

    assert profiler.finished
    text = (tmp_path / "out" / "run_profile.txt").read_text(encoding="utf-8")
    assert "the profiling window never opened; training ended after 1 optimizer step(s)" in text
    assert "never opened" in caplog.text
    # a second finalisation is a no-op
    profiler.abort()
    assert (tmp_path / "out" / "run_profile.txt").read_text(encoding="utf-8") == text
