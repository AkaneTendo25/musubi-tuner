<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess } from '$lib/stores/processes.js';

	function update(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, slider: { ...(c.slider || {}), [key]: value } };
		});
		saveProjectDebounced();
	}

	function updateTarget(index, key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			const targets = [...(c.slider?.targets || [{}])];
			targets[index] = { ...(targets[index] || {}), [key]: value };
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}

	function addTarget() {
		projectConfig.update((c) => {
			if (!c) return c;
			const targets = [...(c.slider?.targets || []), { positive: '', negative: '', target_class: '', weight: 1.0 }];
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}

	function removeTarget(index) {
		projectConfig.update((c) => {
			if (!c?.slider?.targets) return c;
			let targets = c.slider.targets.filter((_, i) => i !== index);
			if (targets.length === 0) targets = [{ positive: '', negative: '', target_class: '', weight: 1.0 }];
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}

	function updateTraining(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, training: { ...(c.training || {}), [key]: value } };
		});
		saveProjectDebounced();
	}

	function updateCaching(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, caching: { ...(c.caching || {}), [key]: value } };
		});
		saveProjectDebounced();
	}

	let targets = $derived($projectConfig?.slider?.targets || [{ positive: '', negative: '', target_class: '', weight: 1.0 }]);
	let sliderStatus = $derived($processStatuses.slider_training || { state: 'idle', exit_code: null });
	let dinoStatus = $derived($processStatuses.cache_dino || { state: 'idle', exit_code: null });
	let dinoLogs = $derived($processLogs.cache_dino || []);
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-5">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Training Techniques</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">Advanced training enhancements and specialized LoRA types.</p>
		</div>

		<!-- CREPA -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">CREPA</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Cross-frame Representation Alignment (arxiv 2506.09229)</div>
					</div>
					<div class="ml-auto">
						<FormToggle checked={$projectConfig?.training?.crepa ?? false} onchange={(e) => updateTraining('crepa', e.target.checked)} />
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Aligns intermediate DiT representations across video frames during fine-tuning, improving temporal consistency. A small projector MLP is trained alongside LoRA.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<!-- Teacher signal mode -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Teacher Signal</div>
					<FormSelect label="Mode" value={$projectConfig?.training?.crepa_mode || 'backbone'} onchange={(e) => updateTraining('crepa_mode', e.target.value)} options={[{value: 'backbone', label: 'Backbone (deeper block)'}, {value: 'dino', label: 'DINOv2 (pre-cached)'}]} tooltip="backbone: deeper transformer block as teacher. dino: pre-cached DINOv2 features (zero VRAM, must cache first on Caching tab)." />

					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField label="Student Block" type="number" value={$projectConfig?.training?.crepa_student_block_idx ?? 16} oninput={(e) => updateTraining('crepa_student_block_idx', Number(e.target.value))} min={0} max={47} tooltip={($projectConfig?.training?.crepa_mode || 'backbone') === 'backbone' ? 'Early block whose hidden states are aligned to the teacher (default 16)' : 'DiT block whose hidden states are projected into DINOv2 space (default 16)'} />
						<FormField label="Teacher Block" type="number" value={$projectConfig?.training?.crepa_teacher_block_idx ?? 32} oninput={(e) => updateTraining('crepa_teacher_block_idx', Number(e.target.value))} min={0} max={47} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'backbone'} tooltip="Deeper block providing the teacher signal (default 32, must be > student)" />
					</div>
					<FormSelect label="DINOv2 Model" value={$projectConfig?.training?.crepa_dino_model || 'dinov2_vitb14'} onchange={(e) => updateTraining('crepa_dino_model', e.target.value)} options={[{value: 'dinov2_vits14', label: 'ViT-S/14 (384d)'}, {value: 'dinov2_vitb14', label: 'ViT-B/14 (768d)'}, {value: 'dinov2_vitl14', label: 'ViT-L/14 (1024d)'}, {value: 'dinov2_vitg14', label: 'ViT-G/14 (1536d)'}]} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'dino'} tooltip="DINOv2 model variant. Must match the model used during caching." />
				</div>

				<!-- Loss parameters -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Parameters</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Lambda" type="number" value={$projectConfig?.training?.crepa_lambda ?? 0.1} oninput={(e) => updateTraining('crepa_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="CREPA loss weight (default 0.1)" />
						<FormField label="Tau" type="number" value={$projectConfig?.training?.crepa_tau ?? 1.0} oninput={(e) => updateTraining('crepa_tau', Number(e.target.value))} step="0.1" min={0.01} tooltip="Temporal neighbor decay factor (default 1.0)" />
						<FormField label="Neighbors" type="number" value={$projectConfig?.training?.crepa_num_neighbors ?? 2} oninput={(e) => updateTraining('crepa_num_neighbors', Number(e.target.value))} min={1} max={8} tooltip="K frames on each side for alignment (default 2)" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormSelect label="Schedule" value={$projectConfig?.training?.crepa_schedule || 'constant'} onchange={(e) => updateTraining('crepa_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="Lambda schedule over training" />
						<FormField label="Warmup Steps" type="number" value={$projectConfig?.training?.crepa_warmup_steps ?? 0} oninput={(e) => updateTraining('crepa_warmup_steps', Number(e.target.value))} min={0} tooltip="Steps before CREPA loss reaches full strength" />
						<div class="flex items-end pb-0.5">
							<FormToggle label="Normalize" checked={$projectConfig?.training?.crepa_normalize ?? true} onchange={(e) => updateTraining('crepa_normalize', e.target.checked)} tooltip="L2-normalize features before similarity computation" />
						</div>
					</div>
				</div>

				<!-- DINOv2 Caching (for CREPA dino mode) -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">DINOv2 Feature Caching</div>
					<p class="text-[10px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Cache DINOv2 features for CREPA dino mode. Run this after latent caching and before training. Uses the DINOv2 model selected above.
					</p>
					<div class="mb-2">
						<FormField label="Batch Size" type="number" value={$projectConfig?.caching?.dino_batch_size ?? 16} oninput={(e) => updateCaching('dino_batch_size', Number(e.target.value))} min={1} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'dino'} tooltip="Frames per DINOv2 forward pass (reduce if OOM)" />
					</div>
					<div class="mb-2">
						<ProcessControls processType="cache_dino" status={dinoStatus} onStart={() => startProcess('cache_dino')} onStop={() => stopProcess('cache_dino')} />
					</div>
					<ProcessConsole lines={dinoLogs} />
					<CommandPanel processType="cache_dino" defaultFilename="cache_dino.bat" />
				</div>
			</div>
		</div>

		<!-- Self-Flow -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Self-Flow</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Dual-timestep noising + EMA-teacher feature alignment regularization</div>
					</div>
					<div class="ml-auto">
						<FormToggle checked={$projectConfig?.training?.self_flow ?? false} onchange={(e) => updateTraining('self_flow', e.target.checked)} />
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Self-Flow regularizes training by aligning student features with an EMA-updated teacher model across different noise levels.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Architecture</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Teacher Mode" type="select" value={$projectConfig?.training?.self_flow_teacher_mode ?? 'base'} oninput={(e) => updateTraining('self_flow_teacher_mode', e.target.value)} options={[{value: 'base', label: 'Base model'}, {value: 'ema', label: 'EMA (all LoRA)'}, {value: 'partial_ema', label: 'Partial EMA (teacher block)'}]} tooltip="base: frozen pretrained model (no VRAM overhead, stronger gap). ema: EMA over all LoRA params. partial_ema: EMA only over teacher block's LoRA params." />
						<FormField label="Student Block" type="number" value={$projectConfig?.training?.self_flow_student_block_idx ?? 16} oninput={(e) => updateTraining('self_flow_student_block_idx', Number(e.target.value))} min={0} max={47} tooltip="Student feature block index (overridden by ratio when set)" />
						<FormField label="Teacher Block" type="number" value={$projectConfig?.training?.self_flow_teacher_block_idx ?? 32} oninput={(e) => updateTraining('self_flow_teacher_block_idx', Number(e.target.value))} min={0} max={47} tooltip="Teacher feature block index (must be > student; overridden by ratio when set)" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Student Ratio" type="number" value={$projectConfig?.training?.self_flow_student_block_ratio ?? 0.3} oninput={(e) => updateTraining('self_flow_student_block_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Ratio-based student block: floor(ratio × depth). Takes priority over block index." />
						<FormField label="Teacher Ratio" type="number" value={$projectConfig?.training?.self_flow_teacher_block_ratio ?? 0.7} oninput={(e) => updateTraining('self_flow_teacher_block_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Ratio-based teacher block: ceil(ratio × depth). Takes priority over block index." />
						<FormField label="Stochastic Range" type="number" value={$projectConfig?.training?.self_flow_student_block_stochastic_range ?? 0} oninput={(e) => updateTraining('self_flow_student_block_stochastic_range', Number(e.target.value))} min={0} max={8} tooltip="Randomly vary student capture block ±N each step. 0 = fixed block. Adds regularization variety but uses a shared projector across depths." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Parameters</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Lambda" type="number" value={$projectConfig?.training?.self_flow_lambda ?? 0.1} oninput={(e) => updateTraining('self_flow_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="Self-Flow base loss weight. Scaled by the same schedule as temporal lambdas." />
						<FormField label="Mask Ratio" type="number" value={$projectConfig?.training?.self_flow_mask_ratio ?? 0.1} oninput={(e) => updateTraining('self_flow_mask_ratio', Number(e.target.value))} step="0.05" min={0} max={0.5} tooltip="Fraction of tokens given the alternate timestep in dual-timestep noising. Valid range [0, 0.5]." />
						<FormField label="Max Loss Cap" type="number" value={$projectConfig?.training?.self_flow_max_loss ?? 0.0} oninput={(e) => updateTraining('self_flow_max_loss', Number(e.target.value))} step="0.01" min={0} placeholder="Disabled" tooltip="Rescale total Self-Flow loss if its magnitude exceeds this value. 0 = disabled. Prevents Self-Flow from dominating the main task loss early in training." />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Momentum" type="number" value={$projectConfig?.training?.self_flow_teacher_momentum ?? 0.999} oninput={(e) => updateTraining('self_flow_teacher_momentum', Number(e.target.value))} step="0.001" min={0} max={1} tooltip="EMA momentum for teacher updates (only used when teacher_mode=ema or partial_ema)" />
						<FormField label="Projector LR" type="number" value={$projectConfig?.training?.self_flow_projector_lr ?? ''} oninput={(e) => updateTraining('self_flow_projector_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Same as LR" step="any" tooltip="Separate LR for projector MLP" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<div class="flex items-end pb-0.5">
							<FormToggle label="Dual Timestep" checked={$projectConfig?.training?.self_flow_dual_timestep ?? true} onchange={(e) => updateTraining('self_flow_dual_timestep', e.target.checked)} tooltip="Sample independent timesteps for student/teacher" />
						</div>
						<div class="flex items-end pb-0.5">
							<FormToggle label="Frame-Level Mask" checked={$projectConfig?.training?.self_flow_frame_level_mask ?? false} onchange={(e) => updateTraining('self_flow_frame_level_mask', e.target.checked)} tooltip="Mask whole frames instead of individual tokens. More semantically meaningful masking for video." />
						</div>
						<div class="flex items-end pb-0.5">
							<FormToggle label="Mask-Focus Loss" checked={$projectConfig?.training?.self_flow_mask_focus_loss ?? false} onchange={(e) => updateTraining('self_flow_mask_focus_loss', e.target.checked)} tooltip="Focus the representation loss only on the masked (higher-noise) tokens. Standard: loss over all tokens." />
						</div>
					</div>
					<div class="grid grid-cols-1 gap-2 mt-2">
						<div class="flex items-end pb-0.5">
							<FormToggle label="Offload Teacher Features" checked={$projectConfig?.training?.self_flow_offload_teacher_features ?? false} onchange={(e) => updateTraining('self_flow_offload_teacher_features', e.target.checked)} tooltip="Offload cached teacher features to CPU to reduce VRAM" />
						</div>
					</div>
				</div>

				<!-- Temporal Consistency -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-1" style="color: var(--text-primary);">Temporal Consistency</div>
					<div class="text-[11px] mb-2" style="color: var(--text-muted);">Frame-neighbor and motion-delta losses to preserve temporal coherence during fine-tuning</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Mode" type="select" value={$projectConfig?.training?.self_flow_temporal_mode ?? 'off'} oninput={(e) => updateTraining('self_flow_temporal_mode', e.target.value)} options={[{value: 'off', label: 'Off'}, {value: 'frame', label: 'Frame'}, {value: 'delta', label: 'Delta'}, {value: 'hybrid', label: 'Hybrid'}]} tooltip="off: disabled, frame: neighbor alignment, delta: motion consistency, hybrid: both" />
						<FormField label="Lambda Temporal" type="number" value={$projectConfig?.training?.self_flow_lambda_temporal ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_temporal', Number(e.target.value))} step="0.01" min={0} tooltip="Loss weight for frame-level temporal alignment (frame/hybrid modes)" />
						<FormField label="Lambda Delta" type="number" value={$projectConfig?.training?.self_flow_lambda_delta ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_delta', Number(e.target.value))} step="0.01" min={0} tooltip="Loss weight for motion delta alignment (delta/hybrid modes)" />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Neighbors" type="number" value={$projectConfig?.training?.self_flow_num_neighbors ?? 2} oninput={(e) => updateTraining('self_flow_num_neighbors', Number(e.target.value))} min={0} max={8} tooltip="Temporal neighbors on each side for frame alignment" />
						<FormField label="Tau" type="number" value={$projectConfig?.training?.self_flow_temporal_tau ?? 1.0} oninput={(e) => updateTraining('self_flow_temporal_tau', Number(e.target.value))} step="0.1" min={0.1} tooltip="Neighbor weight decay factor (higher = slower decay)" />
						<FormField label="Delta Steps" type="number" value={$projectConfig?.training?.self_flow_delta_num_steps ?? 1} oninput={(e) => updateTraining('self_flow_delta_num_steps', Number(e.target.value))} min={1} max={8} tooltip="Multi-step delta: 1 = adjacent frames only" />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Granularity" type="select" value={$projectConfig?.training?.self_flow_temporal_granularity ?? 'frame'} oninput={(e) => updateTraining('self_flow_temporal_granularity', e.target.value)} options={[{value: 'frame', label: 'Frame'}, {value: 'patch', label: 'Patch'}]} tooltip="frame: mean-pooled per frame (fast), patch: per-token spatial (stronger)" />
						<FormField label="Patch Radius" type="number" value={$projectConfig?.training?.self_flow_patch_spatial_radius ?? 0} oninput={(e) => updateTraining('self_flow_patch_spatial_radius', Number(e.target.value))} min={0} max={4} tooltip="Local spatial radius for patch matching (0 = strict position)" />
						<FormField label="Patch Mode" type="select" value={$projectConfig?.training?.self_flow_patch_match_mode ?? 'hard'} oninput={(e) => updateTraining('self_flow_patch_match_mode', e.target.value)} options={[{value: 'hard', label: 'Hard'}, {value: 'soft', label: 'Soft'}]} tooltip="hard: best match in window, soft: softmax-weighted" />
					</div>
					<div class="grid grid-cols-2 gap-2 mb-2">
						<FormField label="Motion Weighting" type="select" value={$projectConfig?.training?.self_flow_motion_weighting ?? 'none'} oninput={(e) => updateTraining('self_flow_motion_weighting', e.target.value)} options={[{value: 'none', label: 'None'}, {value: 'teacher_delta', label: 'Teacher Delta'}]} tooltip="Upweight regions with more motion in teacher features" />
						<FormField label="Motion Strength" type="number" value={$projectConfig?.training?.self_flow_motion_weight_strength ?? 0.0} oninput={(e) => updateTraining('self_flow_motion_weight_strength', Number(e.target.value))} step="0.1" min={0} tooltip="How strongly motion affects per-token weighting" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Schedule" type="select" value={$projectConfig?.training?.self_flow_temporal_schedule ?? 'constant'} oninput={(e) => updateTraining('self_flow_temporal_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="Schedule for all Self-Flow lambdas (lambda_self_flow, lambda_temporal, lambda_delta all scale together)" />
						<FormField label="Warmup Steps" type="number" value={$projectConfig?.training?.self_flow_temporal_warmup_steps ?? 0} oninput={(e) => updateTraining('self_flow_temporal_warmup_steps', Number(e.target.value))} min={0} tooltip="Linear ramp-up before temporal loss reaches full weight" />
						<FormField label="Max Steps" type="number" value={$projectConfig?.training?.self_flow_temporal_max_steps ?? 0} oninput={(e) => updateTraining('self_flow_temporal_max_steps', Number(e.target.value))} min={0} tooltip="Steps at which linear/cosine decay reaches zero (0 = no decay)" />
					</div>
				</div>
			</div>
		</div>

		<!-- HFATO (ViBe) -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">HFATO</div>
						<div class="text-[11px]" style="color: var(--text-muted);">High-Frequency Awareness Training Objective (ViBe, arxiv 2603.23326)</div>
					</div>
					<div class="ml-auto">
						<FormToggle checked={$projectConfig?.training?.hfato ?? false} onchange={(e) => updateTraining('hfato', e.target.checked)} />
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Degrades clean latents via downsample-upsample before noise addition, then trains the model to reconstruct the original clean latents. Enables image-only training without losing video temporal coherence. Use with Relay LoRA workflow for best results.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Parameters</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Scale Factor" type="number" value={$projectConfig?.training?.hfato_scale_factor ?? 0.5} oninput={(e) => updateTraining('hfato_scale_factor', Number(e.target.value))} step="0.05" min={0.05} max={1.0} tooltip="Downsample ratio for spatial degradation. 0.5 = halve each spatial dimension. Lower = more aggressive degradation (default 0.5)" />
						<FormSelect label="Interpolation" value={$projectConfig?.training?.hfato_interpolation || 'bilinear'} onchange={(e) => updateTraining('hfato_interpolation', e.target.value)} options={[{value: 'bilinear', label: 'Bilinear'}, {value: 'nearest', label: 'Nearest'}, {value: 'bicubic', label: 'Bicubic'}]} tooltip="Interpolation mode for downsample-upsample (default bilinear)" />
						<FormField label="Probability" type="number" value={$projectConfig?.training?.hfato_probability ?? 1.0} oninput={(e) => updateTraining('hfato_probability', Number(e.target.value))} step="0.05" min={0.05} max={1.0} tooltip="Probability of applying HFATO per training step. 1.0 = always apply (default 1.0)" />
					</div>
				</div>
			</div>
		</div>

		<!-- Audio Features (shown when mode is av/audio) -->
		{#if ($projectConfig?.training?.ltx2_mode || 'video') !== 'video'}
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Audio Features</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Audio loss balancing, supervision, and regularization</div>
					</div>
				</div>
			</div>

			<div class="p-5 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Balance</div>
					<FormSelect label="Mode" value={$projectConfig?.training?.audio_loss_balance_mode || 'none'} onchange={(e) => updateTraining('audio_loss_balance_mode', e.target.value)} options={[{value: 'none', label: 'None (static weights)'}, {value: 'inv_freq', label: 'Inverse Frequency'}, {value: 'ema_mag', label: 'EMA Magnitude'}]} tooltip="Dynamic audio loss balancing mode" />
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'inv_freq'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField label="Beta" type="number" value={$projectConfig?.training?.audio_loss_balance_beta ?? 0.01} oninput={(e) => updateTraining('audio_loss_balance_beta', Number(e.target.value))} step="0.005" tooltip="EMA update factor" />
						<FormField label="Eps" type="number" value={$projectConfig?.training?.audio_loss_balance_eps ?? 0.05} oninput={(e) => updateTraining('audio_loss_balance_eps', Number(e.target.value))} step="0.01" tooltip="Minimum denominator" />
					</div>
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField label="Min Weight" type="number" value={$projectConfig?.training?.audio_loss_balance_min ?? 0.05} oninput={(e) => updateTraining('audio_loss_balance_min', Number(e.target.value))} step="0.01" tooltip="Minimum audio weight" />
						<FormField label="Max Weight" type="number" value={$projectConfig?.training?.audio_loss_balance_max ?? 4.0} oninput={(e) => updateTraining('audio_loss_balance_max', Number(e.target.value))} step="0.5" tooltip="Maximum audio weight" />
					</div>
					{/if}
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ema_mag'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField label="Target Ratio" type="number" value={$projectConfig?.training?.audio_loss_balance_target_ratio ?? 0.33} oninput={(e) => updateTraining('audio_loss_balance_target_ratio', Number(e.target.value))} step="0.05" tooltip="Target audio/video loss ratio" />
						<FormField label="EMA Decay" type="number" value={$projectConfig?.training?.audio_loss_balance_ema_decay ?? 0.99} oninput={(e) => updateTraining('audio_loss_balance_ema_decay', Number(e.target.value))} step="0.005" tooltip="EMA decay for loss magnitude tracking" />
					</div>
					{/if}
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Options</div>
					<div class="flex flex-wrap gap-x-4 gap-y-1 mb-2">
						<FormToggle label="Independent Timestep" checked={$projectConfig?.training?.independent_audio_timestep ?? false} onchange={(e) => updateTraining('independent_audio_timestep', e.target.checked)} tooltip="Independent timesteps for audio noising" />
						<FormToggle label="Silence Regularizer" checked={$projectConfig?.training?.audio_silence_regularizer ?? false} onchange={(e) => updateTraining('audio_silence_regularizer', e.target.checked)} tooltip="Synthetic silence for missing audio" />
						<FormToggle label="Audio DOP" checked={$projectConfig?.training?.audio_dop ?? false} onchange={(e) => updateTraining('audio_dop', e.target.checked)} tooltip="Preserve base model audio predictions" />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField label="Silence Weight" type="number" value={$projectConfig?.training?.audio_silence_regularizer_weight ?? 1.0} oninput={(e) => updateTraining('audio_silence_regularizer_weight', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.audio_silence_regularizer} tooltip="Weight for silence regularizer loss" />
						<FormField label="DOP Multiplier" type="number" value={$projectConfig?.training?.audio_dop_multiplier ?? 0.5} oninput={(e) => updateTraining('audio_dop_multiplier', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.audio_dop} tooltip="Audio DOP loss multiplier" />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Supervision</div>
					<FormSelect label="Mode" value={$projectConfig?.training?.audio_supervision_mode || 'off'} onchange={(e) => updateTraining('audio_supervision_mode', e.target.value)} options={[{value: 'off', label: 'Off'}, {value: 'warn', label: 'Warn'}, {value: 'error', label: 'Error'}]} tooltip="Monitor AV audio supervision quality" />
					{#if ($projectConfig?.training?.audio_supervision_mode || 'off') !== 'off'}
					<div class="grid grid-cols-3 gap-2 mt-2">
						<FormField label="Warmup" type="number" value={$projectConfig?.training?.audio_supervision_warmup_steps ?? 50} oninput={(e) => updateTraining('audio_supervision_warmup_steps', Number(e.target.value))} min={0} tooltip="Warmup steps" />
						<FormField label="Interval" type="number" value={$projectConfig?.training?.audio_supervision_check_interval ?? 50} oninput={(e) => updateTraining('audio_supervision_check_interval', Number(e.target.value))} min={1} tooltip="Check interval" />
						<FormField label="Min Ratio" type="number" value={$projectConfig?.training?.audio_supervision_min_ratio ?? 0.9} oninput={(e) => updateTraining('audio_supervision_min_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Minimum supervised ratio" />
					</div>
					{/if}
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Buckets & Batch Sampling</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect label="Bucket Strategy" value={$projectConfig?.training?.audio_bucket_strategy || ''} options={[{value:'',label:'Default'},{value:'pad',label:'Pad'},{value:'truncate',label:'Truncate'}]} onchange={(e) => updateTraining('audio_bucket_strategy', e.target.value || null)} tooltip="Audio bucket strategy (pad or truncate)" />
						<FormField label="Bucket Interval" type="number" value={$projectConfig?.training?.audio_bucket_interval ?? ''} oninput={(e) => updateTraining('audio_bucket_interval', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" step="0.1" tooltip="Audio bucket duration interval" />
						<FormField label="Audio Seq Res" type="number" value={$projectConfig?.training?.audio_only_sequence_resolution ?? 64} oninput={(e) => updateTraining('audio_only_sequence_resolution', Number(e.target.value))} min={1} tooltip="Sequence resolution for audio-only mode" />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField label="Min Audio Batches" type="number" value={$projectConfig?.training?.min_audio_batches_per_accum ?? 0} oninput={(e) => updateTraining('min_audio_batches_per_accum', Number(e.target.value))} min={0} tooltip="Min audio batches per accumulation (0=disabled)" />
						<FormField label="Audio Batch Prob" type="number" value={$projectConfig?.training?.audio_batch_probability ?? ''} oninput={(e) => updateTraining('audio_batch_probability', e.target.value ? Number(e.target.value) : null)} placeholder="Random" step="0.1" min={0} max={1} tooltip="Audio batch selection probability" />
					</div>
				</div>
			</div>
		</div>
		{/if}

		<!-- Advanced Timestep -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Advanced Timestep</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Shifted logit-normal sigma sampler and timestep bucketing</div>
					</div>
				</div>
			</div>

			<div class="p-5 pt-2 space-y-2">
				<div class="grid grid-cols-3 gap-2">
					<FormSelect label="Logit Mode" value={$projectConfig?.training?.shifted_logit_mode || ''} options={[{value:'',label:'Auto'},{value:'legacy',label:'Legacy'},{value:'stretched',label:'Stretched'}]} onchange={(e) => updateTraining('shifted_logit_mode', e.target.value || null)} tooltip="legacy: historical behavior, stretched: upstream Mar-2026" />
					<FormField label="Eps" type="number" value={$projectConfig?.training?.shifted_logit_eps ?? 0.001} oninput={(e) => updateTraining('shifted_logit_eps', Number(e.target.value))} step="0.001" tooltip="Numerical epsilon for stretched mode" />
					<FormField label="Uniform Prob" type="number" value={$projectConfig?.training?.shifted_logit_uniform_prob ?? 0.1} oninput={(e) => updateTraining('shifted_logit_uniform_prob', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Uniform fallback probability" />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<div class="flex items-end pb-0.5">
						<FormToggle label="Preserve Distribution Shape" checked={$projectConfig?.training?.preserve_distribution_shape ?? false} onchange={(e) => updateTraining('preserve_distribution_shape', e.target.checked)} tooltip="Use rejection sampling for min/max timestep" />
					</div>
					<FormField label="Timestep Buckets" type="number" value={$projectConfig?.training?.num_timestep_buckets ?? ''} oninput={(e) => updateTraining('num_timestep_buckets', e.target.value ? Number(e.target.value) : null)} placeholder="Off" min={2} tooltip="Stratified timestep sampling buckets" />
				</div>
			</div>
		</div>

		<!-- Preservation & Regularization -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Preservation & Regularization</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Techniques to prevent catastrophic forgetting and maintain model quality</div>
					</div>
				</div>
			</div>

			<div class="p-5 space-y-4">
				<!-- Blank Preservation -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Blank Preservation</span>
						<FormToggle checked={$projectConfig?.training?.blank_preservation ?? false} onchange={(e) => updateTraining('blank_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Regularizes by training on blank (empty) prompts alongside real data, preserving the model's base generation capabilities.
					</p>
					<FormField label="Multiplier" type="number" value={$projectConfig?.training?.blank_preservation_multiplier ?? 1.0} oninput={(e) => updateTraining('blank_preservation_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for blank preservation (default 1.0)" />
				</div>

				<!-- DOP -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">DOP (Differential Output Preservation)</span>
						<FormToggle checked={$projectConfig?.training?.dop ?? false} onchange={(e) => updateTraining('dop', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Preserves the model's output distribution for a specified class by penalizing deviations from the original model during training.
					</p>
					<div class="grid grid-cols-2 gap-2">
						<FormField label="Class Prompt" value={$projectConfig?.training?.dop_class || ''} oninput={(e) => updateTraining('dop_class', e.target.value)} placeholder="woman" tooltip="Target class prompt for output preservation" />
						<FormField label="Multiplier" type="number" value={$projectConfig?.training?.dop_multiplier ?? 1.0} oninput={(e) => updateTraining('dop_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for DOP (default 1.0)" />
					</div>
				</div>

				<!-- Prior Divergence -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Prior Divergence</span>
						<FormToggle checked={$projectConfig?.training?.prior_divergence ?? false} onchange={(e) => updateTraining('prior_divergence', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						KL-divergence regularization that penalizes the trained model from diverging too far from the original pretrained model weights.
					</p>
					<FormField label="Multiplier" type="number" value={$projectConfig?.training?.prior_divergence_multiplier ?? 0.1} oninput={(e) => updateTraining('prior_divergence_multiplier', Number(e.target.value))} step="0.01" min={0} tooltip="KL-divergence regularization strength (default 0.1)" />
				</div>

				<!-- Precaching options -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Precached Preservation</span>
						<FormToggle checked={$projectConfig?.training?.use_precached_preservation ?? false} onchange={(e) => updateTraining('use_precached_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Use pre-cached text encoder outputs for preservation prompts (must be cached during the caching step).
					</p>
					<PathInput label="Cache Dir" value={$projectConfig?.training?.preservation_prompts_cache || ''} oninput={(e) => updateTraining('preservation_prompts_cache', e.target.value)} showFiles tooltip="Directory with cached preservation embeddings" />
				</div>

				<!-- TARP / DCR -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">TARP (Temporal Aligned RoPE Partitioning)</span>
						<FormToggle checked={$projectConfig?.training?.tarp ?? false} onchange={(e) => updateTraining('tarp', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Windowed cross-attention masks restricting each video frame to temporally nearby audio tokens. Requires AV mode. arXiv:2603.18600.
					</p>
					<FormField label="Window Multiplier" type="number" value={$projectConfig?.training?.tarp_window_multiplier ?? 3} oninput={(e) => updateTraining('tarp_window_multiplier', Number(e.target.value))} step="1" min={1} tooltip="Window size = multiplier * (audio_tokens_per_frame). Default 3." />
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">DCR (Dynamic Context Routing)</span>
						<FormToggle checked={$projectConfig?.training?.dcr ?? false} onchange={(e) => updateTraining('dcr', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Per-sample gradient detachment in cross-attention for mixed audio/video batches. Detaches absent-audio and clean-reference streams. Requires AV mode. arXiv:2603.18600.
					</p>
					<FormToggle label="Reference Detach" checked={$projectConfig?.training?.dcr_reference_detach ?? true} onchange={(e) => updateTraining('dcr_reference_detach', e.target.checked)} tooltip="Detach gradients when sigma=0 (clean reference conditioning)" />
				</div>

				<!-- Audio Quality Metrics -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Audio Quality Metrics</span>
						<FormToggle checked={$projectConfig?.training?.audio_metrics ?? false} onchange={(e) => updateTraining('audio_metrics', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Latent-space metrics (FD, temporal coherence, AV sync) run every step at ~0 cost. Mel-space and embedding-space metrics are opt-in.
					</p>
					<div class="flex flex-wrap gap-x-4 gap-y-1 mb-2">
						<FormToggle label="Mel Metrics" checked={$projectConfig?.training?.audio_metrics_mel_metrics ?? false} onchange={(e) => updateTraining('audio_metrics_mel_metrics', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Spectral convergence, MCD, log-spectral distance (periodic, requires VAE decode)" />
						<FormToggle label="CLAP Similarity" checked={$projectConfig?.training?.audio_metrics_clap_similarity ?? false} onchange={(e) => updateTraining('audio_metrics_clap_similarity', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="CLAP audio-text cosine similarity at sampling time" />
						<FormToggle label="AV Onset Alignment" checked={$projectConfig?.training?.audio_metrics_av_onset_alignment ?? false} onchange={(e) => updateTraining('audio_metrics_av_onset_alignment', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Correlation between audio onsets and video motion at sampling time" />
					</div>
					<FormField label="Mel Interval" type="number" value={$projectConfig?.training?.audio_metrics_mel_compute_every ?? 100} oninput={(e) => updateTraining('audio_metrics_mel_compute_every', Number(e.target.value))} step="10" min={1} disabled={!$projectConfig?.training?.audio_metrics_mel_metrics} tooltip="Compute mel metrics every N steps" />
				</div>
			</div>
		</div>

		<!-- Slider LoRA — functional -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<!-- Header -->
			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Slider LoRA</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Train controllable attribute sliders with prompt pairs</div>
					</div>
				</div>
			</div>

			<!-- Config -->
			<div class="p-5 space-y-4">
				<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
					Model, LoRA, optimizer, memory, and output settings are inherited from the Training tab. Only slider-specific settings are shown here.
				</p>

				<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
					<!-- Left: Slider settings -->
					<div class="space-y-3">
						<FormGroup title="Slider Settings">
							<div class="space-y-2 pt-2">
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Steps" type="number" value={$projectConfig?.slider?.max_train_steps ?? 500} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Slider training steps (typically less than full training)" />
									<FormField label="Output Name" value={$projectConfig?.slider?.output_name || 'ltx2_slider'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix for slider LoRA" />
								</div>
								<FormField label="Guidance Strength" type="number" value={$projectConfig?.slider?.guidance_strength ?? 1.0} oninput={(e) => update('guidance_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance strength for text-mode training" />
								<div class="grid grid-cols-3 gap-2">
									<FormField label="Frames" type="number" value={$projectConfig?.slider?.latent_frames ?? 1} oninput={(e) => update('latent_frames', Number(e.target.value))} min={1} tooltip="Latent frames (1=image, >1=video)" />
									<FormField label="Height" type="number" value={$projectConfig?.slider?.latent_height ?? 512} oninput={(e) => update('latent_height', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent height" />
									<FormField label="Width" type="number" value={$projectConfig?.slider?.latent_width ?? 768} oninput={(e) => update('latent_width', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent width" />
								</div>
								<FormField label="Sample Slider Range" value={$projectConfig?.slider?.sample_slider_range || '-2,-1,0,1,2'} oninput={(e) => update('sample_slider_range', e.target.value)} tooltip="Comma-separated multiplier values for preview sampling" />
							</div>
						</FormGroup>
					</div>

					<!-- Right: Targets -->
					<div class="space-y-3">
						<FormGroup title="Slider Targets">
							<div class="space-y-3 pt-2">
								<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
									Define positive/negative prompt pairs that define the slider direction. The LoRA will learn to move between these attributes.
								</p>
								{#each targets as target, i}
									<div class="p-3 space-y-2 relative" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
										<div class="flex items-center justify-between">
											<span class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--accent);">Target #{i + 1}</span>
											{#if targets.length > 1}
												<button
													onclick={() => removeTarget(i)}
													class="px-2 py-0.5 text-[10px] font-medium"
													style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);"
													onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
													onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
												>
													Remove
												</button>
											{/if}
										</div>
										<!-- svelte-ignore a11y_label_has_associated_control -->
										<label class="block">
											<span class="block text-[10px] font-medium mb-0.5" style="color: var(--success);">Positive (+)</span>
											<textarea
												class="w-full text-[11px] px-2 py-1.5 resize-y"
												rows="2"
												value={target.positive || ''}
												oninput={(e) => updateTarget(i, 'positive', e.target.value)}
												placeholder="high quality, sharp, detailed..."
												style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
												onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
												onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
											></textarea>
										</label>
										<!-- svelte-ignore a11y_label_has_associated_control -->
										<label class="block">
											<span class="block text-[10px] font-medium mb-0.5" style="color: var(--danger);">Negative (-)</span>
											<textarea
												class="w-full text-[11px] px-2 py-1.5 resize-y"
												rows="2"
												value={target.negative || ''}
												oninput={(e) => updateTarget(i, 'negative', e.target.value)}
												placeholder="blurry, low quality, soft..."
												style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
												onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
												onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
											></textarea>
										</label>
										<div class="grid grid-cols-2 gap-2">
											<FormField label="Target Class" value={target.target_class || ''} oninput={(e) => updateTarget(i, 'target_class', e.target.value)} placeholder="(all content)" tooltip="Optional: restrict to class" />
											<FormField label="Weight" type="number" value={target.weight ?? 1.0} oninput={(e) => updateTarget(i, 'weight', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for this target" />
										</div>
									</div>
								{/each}
								<button
									onclick={addTarget}
									class="w-full py-1.5 text-[11px] font-medium flex items-center justify-center gap-1"
									style="background: var(--bg-elevated); border: 1px dashed var(--border); color: var(--text-muted); border-radius: var(--radius-sm);"
									onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
									onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
								>
									<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 6v12m6-6H6"/></svg>
									Add Target
								</button>
							</div>
						</FormGroup>
					</div>
				</div>

				<!-- Process Controls -->
				<div class="pt-2">
					<ProcessControls processType="slider_training" status={sliderStatus} onStart={() => startProcess('slider_training')} onStop={() => stopProcess('slider_training')} />
				</div>

				<CommandPanel processType="slider_training" defaultFilename="slider_train.bat" />
			</div>
		</div>
	</div>
{/if}
