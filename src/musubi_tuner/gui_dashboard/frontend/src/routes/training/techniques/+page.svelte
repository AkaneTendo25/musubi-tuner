<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import FieldResetButton from '$lib/components/FieldResetButton.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, preloadLogsIfActive, startLogPolling } from '$lib/stores/processes.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import { onMount } from 'svelte';

	onMount(() => {
		preloadLogsIfActive(['cache_dino', 'slider_training']);
		const logInterval = startLogPolling(['cache_dino', 'slider_training'], 1000);
		return () => clearInterval(logInterval);
	});

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
{:else if !$advancedMode}
	<div class="space-y-4">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Training Techniques</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">This page is only shown in Advanced mode.</p>
		</div>
		<div class="p-5" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="text-[13px] font-semibold mb-1" style="color: var(--text-primary);">Advanced mode required</div>
			<div class="text-[12px]" style="color: var(--text-secondary);">Switch the left sidebar to `Advanced` to access CREPA, Self-Flow, HFATO, slider targets, and the rest of the specialized training controls.</div>
		</div>
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
						<FormToggle fieldPath="training.crepa" checked={$projectConfig?.training?.crepa ?? false} onchange={(e) => updateTraining('crepa', e.target.checked)} />
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
					<FormSelect fieldPath="training.crepa_mode" value={$projectConfig?.training?.crepa_mode || 'backbone'} onchange={(e) => updateTraining('crepa_mode', e.target.value)} options={[{value: 'backbone', label: 'Backbone (deeper block)'}, {value: 'dino', label: 'DINOv2 (pre-cached)'}]} tooltip="backbone: deeper transformer block as teacher. dino: pre-cached DINOv2 features (zero VRAM, must cache first on Caching tab)." />

					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField type="number" fieldPath="training.crepa_student_block_idx" value={$projectConfig?.training?.crepa_student_block_idx ?? 16} oninput={(e) => updateTraining('crepa_student_block_idx', Number(e.target.value))} min={0} max={47} tooltip={($projectConfig?.training?.crepa_mode || 'backbone') === 'backbone' ? 'Early block whose hidden states are aligned to the teacher (default 16)' : 'DiT block whose hidden states are projected into DINOv2 space (default 16)'} />
						<FormField fieldPath="training.crepa_teacher_block_idx" label="Teacher Block" type="number" value={$projectConfig?.training?.crepa_teacher_block_idx ?? 32} oninput={(e) => updateTraining('crepa_teacher_block_idx', Number(e.target.value))} min={0} max={47} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'backbone'} tooltip="Deeper block providing the teacher signal (default 32, must be > student)" />
					</div>
					<FormSelect fieldPath="training.crepa_dino_model" value={$projectConfig?.training?.crepa_dino_model || 'dinov2_vitb14'} onchange={(e) => updateTraining('crepa_dino_model', e.target.value)} options={[{value: 'dinov2_vits14', label: 'ViT-S/14 (384d)'}, {value: 'dinov2_vitb14', label: 'ViT-B/14 (768d)'}, {value: 'dinov2_vitl14', label: 'ViT-L/14 (1024d)'}, {value: 'dinov2_vitg14', label: 'ViT-G/14 (1536d)'}]} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'dino'} tooltip="DINOv2 model variant. Must match the model used during caching." />
				</div>

				<!-- Loss parameters -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Parameters</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField type="number" fieldPath="training.crepa_lambda" value={$projectConfig?.training?.crepa_lambda ?? 0.1} oninput={(e) => updateTraining('crepa_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="CREPA loss weight (default 0.1)" />
						<FormField type="number" fieldPath="training.crepa_tau" value={$projectConfig?.training?.crepa_tau ?? 1.0} oninput={(e) => updateTraining('crepa_tau', Number(e.target.value))} step="0.1" min={0.01} tooltip="Temporal neighbor decay factor (default 1.0)" />
						<FormField type="number" fieldPath="training.crepa_num_neighbors" value={$projectConfig?.training?.crepa_num_neighbors ?? 2} oninput={(e) => updateTraining('crepa_num_neighbors', Number(e.target.value))} min={1} max={8} tooltip="K frames on each side for alignment (default 2)" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormSelect fieldPath="training.crepa_schedule" value={$projectConfig?.training?.crepa_schedule || 'constant'} onchange={(e) => updateTraining('crepa_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="Lambda schedule over training" />
						<FormField type="number" fieldPath="training.crepa_warmup_steps" value={$projectConfig?.training?.crepa_warmup_steps ?? 0} oninput={(e) => updateTraining('crepa_warmup_steps', Number(e.target.value))} min={0} tooltip="Steps before CREPA loss reaches full strength" />
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.crepa_normalize" checked={$projectConfig?.training?.crepa_normalize ?? true} onchange={(e) => updateTraining('crepa_normalize', e.target.checked)} tooltip="L2-normalize features before similarity computation" />
						</div>
					</div>
				</div>
				<FormField fieldPath="training.crepa_args" value={$projectConfig?.training?.crepa_args || ''} oninput={(e) => updateTraining('crepa_args', e.target.value)} placeholder="key=value ..." tooltip="Additional values passed after --crepa_args." />

				<!-- DINOv2 Caching (for CREPA dino mode) -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">DINOv2 Feature Caching</div>
					<p class="text-[10px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Cache DINOv2 features for CREPA dino mode. Run this after latent caching and before training. Uses the DINOv2 model selected above.
					</p>
					<div class="mb-2">
						<FormField type="number" fieldPath="caching.dino_batch_size" value={$projectConfig?.caching?.dino_batch_size ?? 16} oninput={(e) => updateCaching('dino_batch_size', Number(e.target.value))} min={1} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'dino'} tooltip="Frames per DINOv2 forward pass (reduce if OOM)" />
					</div>
					<FormField fieldPath="caching.cache_dino_extra_args" value={$projectConfig?.caching?.cache_dino_extra_args || ''} oninput={(e) => updateCaching('cache_dino_extra_args', e.target.value)} placeholder="--num_workers 4 --debug_mode item" tooltip="Extra arguments appended to the DINO cache command. Use this for any CLI option without a dedicated dashboard control." />
					<div class="mb-2">
						<ProcessControls processType="cache_dino" status={dinoStatus} onStart={() => startProcess('cache_dino')} onStop={() => stopProcess('cache_dino')} />
					</div>
					<ProcessConsole lines={dinoLogs} processType="cache_dino" />
					<CommandPanel processType="cache_dino" defaultFilename="cache_dino.bat" />
				</div>
			</div>
		</div>

		<!-- TREAD -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M4.5 6.75h15M4.5 12h15m-15 5.25h15"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">TREAD</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Sparse token routing during selected transformer layers</div>
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Token routing keeps clean conditioning tokens and routes a subset of noisy video or audio tokens through the selected transformer window. Safe default: off.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-2">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">TREAD</span>
						<FormToggle fieldPath="training.tread" checked={$projectConfig?.training?.tread ?? false} onchange={(e) => updateTraining('tread', e.target.checked)} />
					</div>
					<div class="grid grid-cols-4 gap-2">
						<FormSelect fieldPath="training.tread_target" label="Target" value={$projectConfig?.training?.tread_target || 'video'} onchange={(e) => updateTraining('tread_target', e.target.value)} options={[{value: 'video', label: 'Video'}, {value: 'audio', label: 'Audio'}, {value: 'both', label: 'Both'}]} disabled={!$projectConfig?.training?.tread} tooltip="Token stream routed by TREAD. Video is the safe default; audio and both require an audio-enabled LTX mode." />
						<FormField fieldPath="training.tread_selection_ratio" label="Selection Ratio" type="number" value={$projectConfig?.training?.tread_selection_ratio ?? 0.5} oninput={(e) => updateTraining('tread_selection_ratio', Number(e.target.value))} step="0.05" min={0} max={0.95} disabled={!$projectConfig?.training?.tread} tooltip="Fraction of selected-stream tokens affected by TREAD routing. Default 0.5. Valid range [0, 1)." />
						<FormField fieldPath="training.tread_start_layer_idx" label="Start Layer" type="number" value={$projectConfig?.training?.tread_start_layer_idx ?? ''} oninput={(e) => updateTraining('tread_start_layer_idx', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!$projectConfig?.training?.tread} tooltip="First transformer layer to route. Blank uses 3 for LTX-2.3 and 2 for LTX-2.0." />
						<FormField fieldPath="training.tread_end_layer_idx" label="End Layer" type="number" value={$projectConfig?.training?.tread_end_layer_idx ?? ''} oninput={(e) => updateTraining('tread_end_layer_idx', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!$projectConfig?.training?.tread} tooltip="Last routed layer. Negative values count from the end. Blank uses -4 for LTX-2.3 and -2 for LTX-2.0." />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.tread_args" value={$projectConfig?.training?.tread_args || ''} oninput={(e) => updateTraining('tread_args', e.target.value)} placeholder="target=video selection_ratio=0.5 start_layer_idx=3 end_layer_idx=-4" disabled={!$projectConfig?.training?.tread} tooltip="Additional values passed after --tread_args. Structured fields above override matching values here." />
					</div>
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
						<FormToggle fieldPath="training.self_flow" checked={$projectConfig?.training?.self_flow ?? false} onchange={(e) => updateTraining('self_flow', e.target.checked)} />
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
						<FormSelect fieldPath="training.self_flow_teacher_mode" value={$projectConfig?.training?.self_flow_teacher_mode ?? 'base'} onchange={(e) => updateTraining('self_flow_teacher_mode', e.target.value)} options={[{value: 'base', label: 'Base model'}, {value: 'ema', label: 'EMA (all LoRA)'}, {value: 'partial_ema', label: 'Partial EMA (teacher block)'}]} tooltip="base: frozen pretrained model via zeroed LoRA multipliers, with no EMA shadow params. ema: EMA over all LoRA params. partial_ema: EMA only over teacher block's LoRA params." />
						<FormField type="number" fieldPath="training.self_flow_student_block_idx" value={$projectConfig?.training?.self_flow_student_block_idx ?? 16} oninput={(e) => updateTraining('self_flow_student_block_idx', Number(e.target.value))} min={0} max={47} tooltip="Student feature block index (overridden by ratio when set)" />
						<FormField fieldPath="training.self_flow_teacher_block_idx" label="Teacher Block" type="number" value={$projectConfig?.training?.self_flow_teacher_block_idx ?? 32} oninput={(e) => updateTraining('self_flow_teacher_block_idx', Number(e.target.value))} min={0} max={47} tooltip="Teacher feature block index (must be > student; overridden by ratio when set)" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField type="number" fieldPath="training.self_flow_student_block_ratio" value={$projectConfig?.training?.self_flow_student_block_ratio ?? 0.3} oninput={(e) => updateTraining('self_flow_student_block_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Ratio-based student block: floor(ratio × depth). Takes priority over block index." />
						<FormField type="number" fieldPath="training.self_flow_teacher_block_ratio" value={$projectConfig?.training?.self_flow_teacher_block_ratio ?? 0.7} oninput={(e) => updateTraining('self_flow_teacher_block_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Ratio-based teacher block: ceil(ratio × depth). Takes priority over block index." />
						<FormField type="number" fieldPath="training.self_flow_student_block_stochastic_range" value={$projectConfig?.training?.self_flow_student_block_stochastic_range ?? 0} oninput={(e) => updateTraining('self_flow_student_block_stochastic_range', Number(e.target.value))} min={0} max={8} tooltip="Randomly vary student capture block ±N each step. 0 = fixed block. Adds regularization variety but uses a shared projector across depths." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Parameters</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField type="number" fieldPath="training.self_flow_lambda" value={$projectConfig?.training?.self_flow_lambda ?? 0.1} oninput={(e) => updateTraining('self_flow_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="Self-Flow base loss weight. Scaled by the same schedule as temporal lambdas." />
						<FormField type="number" fieldPath="training.self_flow_mask_ratio" value={$projectConfig?.training?.self_flow_mask_ratio ?? 0.1} oninput={(e) => updateTraining('self_flow_mask_ratio', Number(e.target.value))} step="0.05" min={0} max={0.5} tooltip="Fraction of tokens given the alternate timestep in dual-timestep noising. Valid range [0, 0.5]." />
						<FormField type="number" fieldPath="training.self_flow_max_loss" value={$projectConfig?.training?.self_flow_max_loss ?? 0.0} oninput={(e) => updateTraining('self_flow_max_loss', Number(e.target.value))} step="0.01" min={0} placeholder="Disabled" tooltip="Rescale total Self-Flow loss if its magnitude exceeds this value. 0 = disabled. Prevents Self-Flow from dominating the main task loss early in training." />
					</div>
					<div class="grid grid-cols-4 gap-2 mb-2">
						<FormField type="number" fieldPath="training.self_flow_teacher_momentum" value={$projectConfig?.training?.self_flow_teacher_momentum ?? 0.999} oninput={(e) => updateTraining('self_flow_teacher_momentum', Number(e.target.value))} step="0.001" min={0} max={1} tooltip="EMA momentum for teacher updates (only used when teacher_mode=ema or partial_ema)" />
						<FormField type="number" fieldPath="training.self_flow_projector_lr" value={$projectConfig?.training?.self_flow_projector_lr ?? ''} oninput={(e) => updateTraining('self_flow_projector_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Same as LR" step="any" tooltip="Separate LR for projector MLP" />
						<FormSelect fieldPath="training.self_flow_projector_activation" value={$projectConfig?.training?.self_flow_projector_activation ?? 'silu'} onchange={(e) => updateTraining('self_flow_projector_activation', e.target.value)} options={['silu', 'gelu']} tooltip="Projector activation for Self-Flow." />
						<FormField type="number" fieldPath="training.self_flow_lambda_audio" value={$projectConfig?.training?.self_flow_lambda_audio ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_audio', Number(e.target.value))} step="0.01" min={0} tooltip="Optional Self-Flow audio loss weight." />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_dual_timestep" checked={$projectConfig?.training?.self_flow_dual_timestep ?? true} onchange={(e) => updateTraining('self_flow_dual_timestep', e.target.checked)} tooltip="Sample independent timesteps for student/teacher" />
						</div>
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_frame_level_mask" checked={$projectConfig?.training?.self_flow_frame_level_mask ?? false} onchange={(e) => updateTraining('self_flow_frame_level_mask', e.target.checked)} tooltip="Mask whole frames instead of individual tokens. More semantically meaningful masking for video." />
						</div>
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_mask_focus_loss" checked={$projectConfig?.training?.self_flow_mask_focus_loss ?? false} onchange={(e) => updateTraining('self_flow_mask_focus_loss', e.target.checked)} tooltip="Focus the representation loss only on the masked (higher-noise) tokens. Standard: loss over all tokens." />
						</div>
					</div>
					<div class="grid grid-cols-1 gap-2 mt-2">
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_offload_teacher_features" checked={$projectConfig?.training?.self_flow_offload_teacher_features ?? false} onchange={(e) => updateTraining('self_flow_offload_teacher_features', e.target.checked)} tooltip="Offload cached teacher features to CPU to reduce VRAM" />
						</div>
					</div>
				</div>

				<!-- Temporal Consistency -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-1" style="color: var(--text-primary);">Temporal Consistency</div>
					<div class="text-[11px] mb-2" style="color: var(--text-muted);">Frame-neighbor and motion-delta losses to preserve temporal coherence during fine-tuning</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_temporal_mode" value={$projectConfig?.training?.self_flow_temporal_mode ?? 'off'} onchange={(e) => updateTraining('self_flow_temporal_mode', e.target.value)} options={[{value: 'off', label: 'Off'}, {value: 'frame', label: 'Frame'}, {value: 'delta', label: 'Delta'}, {value: 'hybrid', label: 'Hybrid'}]} tooltip="off: disabled, frame: neighbor alignment, delta: motion consistency, hybrid: both" />
						<FormField type="number" fieldPath="training.self_flow_lambda_temporal" value={$projectConfig?.training?.self_flow_lambda_temporal ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_temporal', Number(e.target.value))} step="0.01" min={0} tooltip="Loss weight for frame-level temporal alignment (frame/hybrid modes)" />
						<FormField type="number" fieldPath="training.self_flow_lambda_delta" value={$projectConfig?.training?.self_flow_lambda_delta ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_delta', Number(e.target.value))} step="0.01" min={0} tooltip="Loss weight for motion delta alignment (delta/hybrid modes)" />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField type="number" fieldPath="training.self_flow_num_neighbors" value={$projectConfig?.training?.self_flow_num_neighbors ?? 2} oninput={(e) => updateTraining('self_flow_num_neighbors', Number(e.target.value))} min={0} max={8} tooltip="Temporal neighbors on each side for frame alignment" />
						<FormField type="number" fieldPath="training.self_flow_temporal_tau" value={$projectConfig?.training?.self_flow_temporal_tau ?? 1.0} oninput={(e) => updateTraining('self_flow_temporal_tau', Number(e.target.value))} step="0.1" min={0.1} tooltip="Neighbor weight decay factor (higher = slower decay)" />
						<FormField type="number" fieldPath="training.self_flow_delta_num_steps" value={$projectConfig?.training?.self_flow_delta_num_steps ?? 1} oninput={(e) => updateTraining('self_flow_delta_num_steps', Number(e.target.value))} min={1} max={8} tooltip="Multi-step delta: 1 = adjacent frames only" />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_temporal_granularity" value={$projectConfig?.training?.self_flow_temporal_granularity ?? 'frame'} onchange={(e) => updateTraining('self_flow_temporal_granularity', e.target.value)} options={[{value: 'frame', label: 'Frame'}, {value: 'patch', label: 'Patch'}]} tooltip="frame: mean-pooled per frame (fast), patch: per-token spatial (stronger)" />
						<FormField type="number" fieldPath="training.self_flow_patch_spatial_radius" value={$projectConfig?.training?.self_flow_patch_spatial_radius ?? 0} oninput={(e) => updateTraining('self_flow_patch_spatial_radius', Number(e.target.value))} min={0} max={4} tooltip="Local spatial radius for patch matching (0 = strict position)" />
						<FormSelect fieldPath="training.self_flow_patch_match_mode" value={$projectConfig?.training?.self_flow_patch_match_mode ?? 'hard'} onchange={(e) => updateTraining('self_flow_patch_match_mode', e.target.value)} options={[{value: 'hard', label: 'Hard'}, {value: 'soft', label: 'Soft'}]} tooltip="hard: best match in window, soft: softmax-weighted" />
					</div>
					<div class="grid grid-cols-2 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_motion_weighting" value={$projectConfig?.training?.self_flow_motion_weighting ?? 'none'} onchange={(e) => updateTraining('self_flow_motion_weighting', e.target.value)} options={[{value: 'none', label: 'None'}, {value: 'teacher_delta', label: 'Teacher Delta'}]} tooltip="Upweight regions with more motion in teacher features" />
						<FormField type="number" fieldPath="training.self_flow_motion_weight_strength" value={$projectConfig?.training?.self_flow_motion_weight_strength ?? 0.0} oninput={(e) => updateTraining('self_flow_motion_weight_strength', Number(e.target.value))} step="0.1" min={0} tooltip="How strongly motion affects per-token weighting" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormSelect fieldPath="training.self_flow_temporal_schedule" value={$projectConfig?.training?.self_flow_temporal_schedule ?? 'constant'} onchange={(e) => updateTraining('self_flow_temporal_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="Schedule for all Self-Flow lambdas (lambda_self_flow, lambda_temporal, lambda_delta all scale together)" />
						<FormField type="number" fieldPath="training.self_flow_temporal_warmup_steps" value={$projectConfig?.training?.self_flow_temporal_warmup_steps ?? 0} oninput={(e) => updateTraining('self_flow_temporal_warmup_steps', Number(e.target.value))} min={0} tooltip="Linear ramp-up before temporal loss reaches full weight" />
						<FormField type="number" fieldPath="training.self_flow_temporal_max_steps" value={$projectConfig?.training?.self_flow_temporal_max_steps ?? 0} oninput={(e) => updateTraining('self_flow_temporal_max_steps', Number(e.target.value))} min={0} tooltip="Steps at which linear/cosine decay reaches zero (0 = no decay)" />
					</div>
				</div>
				<FormField fieldPath="training.self_flow_args" value={$projectConfig?.training?.self_flow_args || ''} oninput={(e) => updateTraining('self_flow_args', e.target.value)} placeholder="key=value ..." tooltip="Additional values passed after --self_flow_args." />
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
						<FormToggle fieldPath="training.hfato" checked={$projectConfig?.training?.hfato ?? false} onchange={(e) => updateTraining('hfato', e.target.checked)} />
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
						<FormField type="number" fieldPath="training.hfato_scale_factor" value={$projectConfig?.training?.hfato_scale_factor ?? 0.5} oninput={(e) => updateTraining('hfato_scale_factor', Number(e.target.value))} step="0.05" min={0.05} max={1.0} tooltip="Downsample ratio for spatial degradation. 0.5 = halve each spatial dimension. Lower = more aggressive degradation (default 0.5)" />
						<FormSelect fieldPath="training.hfato_interpolation" value={$projectConfig?.training?.hfato_interpolation || 'bilinear'} onchange={(e) => updateTraining('hfato_interpolation', e.target.value)} options={[{value: 'bilinear', label: 'Bilinear'}, {value: 'nearest', label: 'Nearest'}, {value: 'bicubic', label: 'Bicubic'}]} tooltip="Interpolation mode for downsample-upsample (default bilinear)" />
						<FormField type="number" fieldPath="training.hfato_probability" value={$projectConfig?.training?.hfato_probability ?? 1.0} oninput={(e) => updateTraining('hfato_probability', Number(e.target.value))} step="0.05" min={0.05} max={1.0} tooltip="Probability of applying HFATO per training step. 1.0 = always apply (default 1.0)" />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.hfato_args" value={$projectConfig?.training?.hfato_args || ''} oninput={(e) => updateTraining('hfato_args', e.target.value)} placeholder="key=value ..." tooltip="Additional values passed after --hfato_args." />
					</div>
				</div>
			</div>
		</div>

		<!-- Latent Temporal Objectives -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M4.5 12.75l6-6 4.5 4.5 4.5-4.5M4.5 17.25l6-6 4.5 4.5 4.5-4.5"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Latent Temporal Objectives</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Clean-latent motion weighting and predicted-latent derivative loss</div>
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Training-only terms: per-frame denoising weights from clean latent deltas, plus optional derivative matching on predicted x0 or velocity.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-2">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Latent Motion Weighting</span>
						<FormToggle fieldPath="training.latent_temporal_weighting" checked={$projectConfig?.training?.latent_temporal_weighting ?? false} onchange={(e) => updateTraining('latent_temporal_weighting', e.target.checked)} />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField type="number" fieldPath="training.latent_temporal_weighting_alpha" value={$projectConfig?.training?.latent_temporal_weighting_alpha ?? 0.5} oninput={(e) => updateTraining('latent_temporal_weighting_alpha', Number(e.target.value))} step="0.05" min={0} disabled={!$projectConfig?.training?.latent_temporal_weighting} tooltip="Strength of clean-latent motion weighting. Start lower for full fine-tuning." />
						<FormSelect fieldPath="training.latent_temporal_weighting_mode" value={$projectConfig?.training?.latent_temporal_weighting_mode || 'log'} onchange={(e) => updateTraining('latent_temporal_weighting_mode', e.target.value)} options={[{value: 'log', label: 'Log'}, {value: 'linear', label: 'Linear'}]} disabled={!$projectConfig?.training?.latent_temporal_weighting} tooltip="How frame-to-frame latent discrepancy is converted to motion scores." />
						<FormSelect fieldPath="training.latent_temporal_weighting_normalize" value={$projectConfig?.training?.latent_temporal_weighting_normalize || 'mean'} onchange={(e) => updateTraining('latent_temporal_weighting_normalize', e.target.value)} options={[{value: 'mean', label: 'Mean'}, {value: 'max', label: 'Max'}, {value: 'none', label: 'None'}]} disabled={!$projectConfig?.training?.latent_temporal_weighting} tooltip="Normalization applied before turning motion scores into loss weights." />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField type="number" fieldPath="training.latent_temporal_weighting_clip_min" value={$projectConfig?.training?.latent_temporal_weighting_clip_min ?? 0.5} oninput={(e) => updateTraining('latent_temporal_weighting_clip_min', Number(e.target.value))} step="0.05" min={0.01} disabled={!$projectConfig?.training?.latent_temporal_weighting} tooltip="Lower clamp before final mean rescale." />
						<FormField type="number" fieldPath="training.latent_temporal_weighting_clip_max" value={$projectConfig?.training?.latent_temporal_weighting_clip_max ?? 2.0} oninput={(e) => updateTraining('latent_temporal_weighting_clip_max', Number(e.target.value))} step="0.05" min={0.01} disabled={!$projectConfig?.training?.latent_temporal_weighting} tooltip="Upper clamp before final mean rescale." />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.latent_temporal_weighting_args" value={$projectConfig?.training?.latent_temporal_weighting_args || ''} oninput={(e) => updateTraining('latent_temporal_weighting_args', e.target.value)} placeholder="alpha=0.5 clip_max=2.0" disabled={!$projectConfig?.training?.latent_temporal_weighting} tooltip="Additional values passed after --latent_temporal_weighting_args." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-2">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Predicted Latent Delta Loss</span>
						<FormToggle fieldPath="training.latent_delta_loss" checked={$projectConfig?.training?.latent_delta_loss ?? false} onchange={(e) => updateTraining('latent_delta_loss', e.target.checked)} />
					</div>
					<div class="grid grid-cols-4 gap-2 mb-2">
						<FormField type="number" fieldPath="training.latent_delta_loss_weight" value={$projectConfig?.training?.latent_delta_loss_weight ?? 0.03} oninput={(e) => updateTraining('latent_delta_loss_weight', Number(e.target.value))} step="0.005" min={0} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Extra latent temporal loss weight. Use 0.01-0.05 as a first range." />
						<FormSelect fieldPath="training.latent_delta_loss_order" value={$projectConfig?.training?.latent_delta_loss_order || '1'} onchange={(e) => updateTraining('latent_delta_loss_order', e.target.value)} options={[{value: '1', label: 'Delta'}, {value: '2', label: 'Accel'}, {value: '1+2', label: 'Delta + Accel'}]} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="First-order delta, second-order acceleration, or both." />
						<FormSelect fieldPath="training.latent_delta_loss_target" value={$projectConfig?.training?.latent_delta_loss_target || 'x0'} onchange={(e) => updateTraining('latent_delta_loss_target', e.target.value)} options={[{value: 'x0', label: 'x0'}, {value: 'velocity', label: 'Velocity'}]} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Latent quantity whose temporal derivatives are matched." />
						<FormSelect fieldPath="training.latent_delta_loss_type" value={$projectConfig?.training?.latent_delta_loss_type || 'mse'} onchange={(e) => updateTraining('latent_delta_loss_type', e.target.value)} options={[{value: 'mse', label: 'MSE'}, {value: 'l1', label: 'L1'}, {value: 'huber', label: 'Huber'}, {value: 'smooth_l1', label: 'Smooth L1'}]} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Loss function for temporal derivative matching." />
					</div>
					<div class="grid grid-cols-4 gap-2">
						<FormField type="number" fieldPath="training.latent_delta_loss_sigma_min" value={$projectConfig?.training?.latent_delta_loss_sigma_min ?? 0.05} oninput={(e) => updateTraining('latent_delta_loss_sigma_min', Number(e.target.value))} step="0.01" min={0} max={1} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Minimum sigma where the extra loss is active." />
						<FormField type="number" fieldPath="training.latent_delta_loss_sigma_max" value={$projectConfig?.training?.latent_delta_loss_sigma_max ?? 0.85} oninput={(e) => updateTraining('latent_delta_loss_sigma_max', Number(e.target.value))} step="0.01" min={0} max={1} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Maximum sigma where the extra loss is active." />
						<FormField type="number" fieldPath="training.latent_delta_loss_second_order_weight" value={$projectConfig?.training?.latent_delta_loss_second_order_weight ?? 0.5} oninput={(e) => updateTraining('latent_delta_loss_second_order_weight', Number(e.target.value))} step="0.05" min={0} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Relative weight for second-order acceleration when enabled." />
						<FormField type="number" fieldPath="training.latent_delta_loss_huber_delta" value={$projectConfig?.training?.latent_delta_loss_huber_delta ?? 1.0} oninput={(e) => updateTraining('latent_delta_loss_huber_delta', Number(e.target.value))} step="0.1" min={0.01} disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Huber beta when using huber or smooth_l1." />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.latent_delta_loss_args" value={$projectConfig?.training?.latent_delta_loss_args || ''} oninput={(e) => updateTraining('latent_delta_loss_args', e.target.value)} placeholder="weight=0.03 order=1 target=x0" disabled={!$projectConfig?.training?.latent_delta_loss} tooltip="Additional values passed after --latent_delta_loss_args." />
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
					<FormSelect fieldPath="training.audio_loss_balance_mode" value={$projectConfig?.training?.audio_loss_balance_mode || 'none'} onchange={(e) => updateTraining('audio_loss_balance_mode', e.target.value)} options={[{value: 'none', label: 'None (static weights)'}, {value: 'inv_freq', label: 'Inverse Frequency'}, {value: 'ema_mag', label: 'EMA Magnitude'}, {value: 'uncertainty', label: 'Uncertainty'}, {value: 'ogm_ge', label: 'OGM-GE'}]} tooltip="Dynamic audio loss balancing mode" />
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField type="number" fieldPath="training.audio_loss_balance_ema_init" value={$projectConfig?.training?.audio_loss_balance_ema_init ?? 1.0} oninput={(e) => updateTraining('audio_loss_balance_ema_init', Number(e.target.value))} step="0.1" min={0} tooltip="Initial EMA value for audio-loss balancing." />
						{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'uncertainty'}
							<FormField type="number" fieldPath="training.uncertainty_lr" value={$projectConfig?.training?.uncertainty_lr ?? ''} oninput={(e) => updateTraining('uncertainty_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="any" tooltip="Learning rate for uncertainty-based balancing." />
						{:else if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ogm_ge'}
							<FormField type="number" fieldPath="training.ogm_ge_alpha" value={$projectConfig?.training?.ogm_ge_alpha ?? 0.3} oninput={(e) => updateTraining('ogm_ge_alpha', Number(e.target.value))} step="0.05" min={0} tooltip="Alpha parameter for OGM-GE balancing." />
						{/if}
					</div>
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'inv_freq'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField type="number" fieldPath="training.audio_loss_balance_beta" value={$projectConfig?.training?.audio_loss_balance_beta ?? 0.01} oninput={(e) => updateTraining('audio_loss_balance_beta', Number(e.target.value))} step="0.005" tooltip="EMA update factor" />
						<FormField type="number" fieldPath="training.audio_loss_balance_eps" value={$projectConfig?.training?.audio_loss_balance_eps ?? 0.05} oninput={(e) => updateTraining('audio_loss_balance_eps', Number(e.target.value))} step="0.01" tooltip="Minimum denominator" />
					</div>
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField type="number" fieldPath="training.audio_loss_balance_min" value={$projectConfig?.training?.audio_loss_balance_min ?? 0.05} oninput={(e) => updateTraining('audio_loss_balance_min', Number(e.target.value))} step="0.01" tooltip="Minimum audio weight" />
						<FormField type="number" fieldPath="training.audio_loss_balance_max" value={$projectConfig?.training?.audio_loss_balance_max ?? 4.0} oninput={(e) => updateTraining('audio_loss_balance_max', Number(e.target.value))} step="0.5" tooltip="Maximum audio weight" />
					</div>
					{/if}
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ema_mag'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField type="number" fieldPath="training.audio_loss_balance_target_ratio" value={$projectConfig?.training?.audio_loss_balance_target_ratio ?? 0.33} oninput={(e) => updateTraining('audio_loss_balance_target_ratio', Number(e.target.value))} step="0.05" tooltip="Target audio/video loss ratio" />
						<FormField type="number" fieldPath="training.audio_loss_balance_ema_decay" value={$projectConfig?.training?.audio_loss_balance_ema_decay ?? 0.99} oninput={(e) => updateTraining('audio_loss_balance_ema_decay', Number(e.target.value))} step="0.005" tooltip="EMA decay for loss magnitude tracking" />
					</div>
					{/if}
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ogm_ge'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField type="number" fieldPath="training.ogm_ge_noise_std" value={$projectConfig?.training?.ogm_ge_noise_std ?? 0.0} oninput={(e) => updateTraining('ogm_ge_noise_std', Number(e.target.value))} step="0.01" min={0} tooltip="Gaussian noise standard deviation for OGM-GE." />
						<div></div>
					</div>
					{/if}
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Options</div>
					<div class="grid grid-cols-3 gap-x-4 gap-y-1 mb-2">
						<FormToggle fieldPath="training.independent_audio_timestep" checked={$projectConfig?.training?.independent_audio_timestep ?? false} onchange={(e) => updateTraining('independent_audio_timestep', e.target.checked)} tooltip="Independent timesteps for audio noising" />
						<FormToggle fieldPath="training.audio_silence_regularizer" checked={$projectConfig?.training?.audio_silence_regularizer ?? false} onchange={(e) => updateTraining('audio_silence_regularizer', e.target.checked)} tooltip="Synthetic silence for missing audio" />
						<FormToggle fieldPath="training.audio_dop" checked={$projectConfig?.training?.audio_dop ?? false} onchange={(e) => updateTraining('audio_dop', e.target.checked)} tooltip="Preserve base model audio predictions" />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField type="number" fieldPath="training.audio_silence_regularizer_weight" value={$projectConfig?.training?.audio_silence_regularizer_weight ?? 1.0} oninput={(e) => updateTraining('audio_silence_regularizer_weight', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.audio_silence_regularizer} tooltip="Weight for silence regularizer loss" />
						<FormField type="number" fieldPath="training.audio_dop_multiplier" value={$projectConfig?.training?.audio_dop_multiplier ?? 0.5} oninput={(e) => updateTraining('audio_dop_multiplier', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.audio_dop} tooltip="Audio DOP loss multiplier" />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.audio_dop_args" value={$projectConfig?.training?.audio_dop_args || ''} oninput={(e) => updateTraining('audio_dop_args', e.target.value)} placeholder="multiplier=0.5" disabled={!$projectConfig?.training?.audio_dop} tooltip="Additional values passed after --audio_dop_args." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Supervision</div>
					<FormSelect fieldPath="training.audio_supervision_mode" value={$projectConfig?.training?.audio_supervision_mode || 'off'} onchange={(e) => updateTraining('audio_supervision_mode', e.target.value)} options={[{value: 'off', label: 'Off'}, {value: 'warn', label: 'Warn'}, {value: 'error', label: 'Error'}]} tooltip="Monitor AV audio supervision quality" />
					{#if ($projectConfig?.training?.audio_supervision_mode || 'off') !== 'off'}
					<div class="grid grid-cols-3 gap-2 mt-2">
						<FormField type="number" fieldPath="training.audio_supervision_warmup_steps" value={$projectConfig?.training?.audio_supervision_warmup_steps ?? 50} oninput={(e) => updateTraining('audio_supervision_warmup_steps', Number(e.target.value))} min={0} tooltip="Warmup steps" />
						<FormField type="number" fieldPath="training.audio_supervision_check_interval" value={$projectConfig?.training?.audio_supervision_check_interval ?? 50} oninput={(e) => updateTraining('audio_supervision_check_interval', Number(e.target.value))} min={1} tooltip="Check interval" />
						<FormField type="number" fieldPath="training.audio_supervision_min_ratio" value={$projectConfig?.training?.audio_supervision_min_ratio ?? 0.9} oninput={(e) => updateTraining('audio_supervision_min_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Minimum supervised ratio" />
					</div>
					{/if}
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Buckets & Batch Sampling</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.audio_bucket_strategy" value={$projectConfig?.training?.audio_bucket_strategy || ''} options={[{value:'',label:'Default'},{value:'pad',label:'Pad'},{value:'truncate',label:'Truncate'}]} onchange={(e) => updateTraining('audio_bucket_strategy', e.target.value || null)} tooltip="Audio bucket strategy (pad or truncate)" />
						<FormField type="number" fieldPath="training.audio_bucket_interval" value={$projectConfig?.training?.audio_bucket_interval ?? ''} oninput={(e) => updateTraining('audio_bucket_interval', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" step="0.1" tooltip="Audio bucket duration interval" />
						<FormField type="number" fieldPath="training.audio_only_sequence_resolution" value={$projectConfig?.training?.audio_only_sequence_resolution ?? 64} oninput={(e) => updateTraining('audio_only_sequence_resolution', Number(e.target.value))} min={1} tooltip="Sequence resolution for audio-only mode" />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField type="number" fieldPath="training.min_audio_batches_per_accum" value={$projectConfig?.training?.min_audio_batches_per_accum ?? 0} oninput={(e) => updateTraining('min_audio_batches_per_accum', Number(e.target.value))} min={0} tooltip="Min audio batches per accumulation (0=disabled)" />
						<FormField type="number" fieldPath="training.audio_batch_probability" value={$projectConfig?.training?.audio_batch_probability ?? ''} oninput={(e) => updateTraining('audio_batch_probability', e.target.value ? Number(e.target.value) : null)} placeholder="Random" step="0.1" min={0} max={1} tooltip="Audio batch selection probability" />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Cross-Token Sync & Modality Freeze</div>
					<div class="grid grid-cols-2 gap-2 mb-2">
						<FormField type="number" fieldPath="training.cts_lambda_video_driven" value={$projectConfig?.training?.cts_lambda_video_driven ?? 0.0} oninput={(e) => updateTraining('cts_lambda_video_driven', Number(e.target.value))} step="0.01" min={0} tooltip="Cross-token sync weight driven by video tokens." />
						<FormField type="number" fieldPath="training.cts_lambda_audio_driven" value={$projectConfig?.training?.cts_lambda_audio_driven ?? 0.0} oninput={(e) => updateTraining('cts_lambda_audio_driven', Number(e.target.value))} step="0.01" min={0} tooltip="Cross-token sync weight driven by audio tokens." />
					</div>
					<div class="grid grid-cols-4 gap-2">
						<FormField type="number" fieldPath="training.modality_freeze_check_interval" value={$projectConfig?.training?.modality_freeze_check_interval ?? 0} oninput={(e) => updateTraining('modality_freeze_check_interval', Number(e.target.value))} min={0} tooltip="0 disables automatic modality freezing." />
						<FormField type="number" fieldPath="training.modality_freeze_ratio_threshold" value={$projectConfig?.training?.modality_freeze_ratio_threshold ?? 0.5} oninput={(e) => updateTraining('modality_freeze_ratio_threshold', Number(e.target.value))} step="0.05" min={0} tooltip="Video/audio loss ratio threshold for freezing." />
						<FormField type="number" fieldPath="training.modality_freeze_warmup_steps" value={$projectConfig?.training?.modality_freeze_warmup_steps ?? 100} oninput={(e) => updateTraining('modality_freeze_warmup_steps', Number(e.target.value))} min={0} tooltip="Warmup steps before freeze decisions are allowed." />
						<FormField type="number" fieldPath="training.modality_freeze_ema_decay" value={$projectConfig?.training?.modality_freeze_ema_decay ?? 0.99} oninput={(e) => updateTraining('modality_freeze_ema_decay', Number(e.target.value))} step="0.005" min={0} max={1} tooltip="EMA decay for modality-freeze loss tracking." />
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
					<FormSelect fieldPath="training.shifted_logit_mode" value={$projectConfig?.training?.shifted_logit_mode || ''} options={[{value:'',label:'Auto'},{value:'legacy',label:'Legacy'},{value:'stretched',label:'Stretched'}]} onchange={(e) => updateTraining('shifted_logit_mode', e.target.value || null)} tooltip="legacy: historical behavior, stretched: percentile stretch behavior" />
					<FormField type="number" fieldPath="training.shifted_logit_eps" value={$projectConfig?.training?.shifted_logit_eps ?? 0.001} oninput={(e) => updateTraining('shifted_logit_eps', Number(e.target.value))} step="0.001" tooltip="Numerical epsilon for stretched mode" />
					<FormField type="number" fieldPath="training.shifted_logit_uniform_prob" value={$projectConfig?.training?.shifted_logit_uniform_prob ?? 0.1} oninput={(e) => updateTraining('shifted_logit_uniform_prob', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Uniform fallback probability" />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<div class="flex items-end pb-0.5">
						<FormToggle fieldPath="training.preserve_distribution_shape" checked={$projectConfig?.training?.preserve_distribution_shape ?? false} onchange={(e) => updateTraining('preserve_distribution_shape', e.target.checked)} tooltip="Use rejection sampling for min/max timestep" />
					</div>
					<FormField type="number" fieldPath="training.num_timestep_buckets" value={$projectConfig?.training?.num_timestep_buckets ?? ''} oninput={(e) => updateTraining('num_timestep_buckets', e.target.value ? Number(e.target.value) : null)} placeholder="Off" min={2} tooltip="Stratified timestep sampling buckets" />
				</div>
				<div class="grid grid-cols-3 gap-2">
					<div class="flex items-end pb-0.5">
						<FormToggle fieldPath="training.shifted_logit_clamp_auto_shift" checked={$projectConfig?.training?.shifted_logit_clamp_auto_shift ?? false} onchange={(e) => updateTraining('shifted_logit_clamp_auto_shift', e.target.checked)} tooltip="Clamp auto-computed shifted-logit shifts to the min/max shift bounds." />
					</div>
					<FormField type="number" fieldPath="training.shifted_logit_min_shift" value={$projectConfig?.training?.shifted_logit_min_shift ?? 0.95} oninput={(e) => updateTraining('shifted_logit_min_shift', Number(e.target.value))} step="0.01" tooltip="Lower clamp bound for auto-computed shifted-logit shifts." />
					<FormField type="number" fieldPath="training.shifted_logit_max_shift" value={$projectConfig?.training?.shifted_logit_max_shift ?? 2.05} oninput={(e) => updateTraining('shifted_logit_max_shift', Number(e.target.value))} step="0.01" tooltip="Upper clamp bound for auto-computed shifted-logit shifts." />
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
						<FormToggle fieldPath="training.blank_preservation" checked={$projectConfig?.training?.blank_preservation ?? false} onchange={(e) => updateTraining('blank_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Regularizes by training on blank (empty) prompts alongside real data, preserving the model's base generation capabilities.
					</p>
					<FormField type="number" fieldPath="training.blank_preservation_multiplier" value={$projectConfig?.training?.blank_preservation_multiplier ?? 1.0} oninput={(e) => updateTraining('blank_preservation_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for blank preservation (default 1.0)" />
					<div class="mt-2">
						<FormField fieldPath="training.blank_preservation_args" value={$projectConfig?.training?.blank_preservation_args || ''} oninput={(e) => updateTraining('blank_preservation_args', e.target.value)} placeholder="multiplier=1.0" tooltip="Additional values passed after --blank_preservation_args." />
					</div>
				</div>

				<!-- DOP -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">DOP (Differential Output Preservation)</span>
						<FormToggle fieldPath="training.dop" checked={$projectConfig?.training?.dop ?? false} onchange={(e) => updateTraining('dop', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Preserves the model's output distribution for a specified class by penalizing deviations from the original model during training.
					</p>
					<div class="grid grid-cols-2 gap-2">
						<FormField fieldPath="training.dop_class" value={$projectConfig?.training?.dop_class || ''} oninput={(e) => updateTraining('dop_class', e.target.value)} placeholder="woman" tooltip="Target class prompt for output preservation" />
						<FormField type="number" fieldPath="training.dop_multiplier" value={$projectConfig?.training?.dop_multiplier ?? 1.0} oninput={(e) => updateTraining('dop_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for DOP (default 1.0)" />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.dop_args" value={$projectConfig?.training?.dop_args || ''} oninput={(e) => updateTraining('dop_args', e.target.value)} placeholder="class=person multiplier=1.0" tooltip="Additional values passed after --dop_args." />
					</div>
				</div>

				<!-- Prior Divergence -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Prior Divergence</span>
						<FormToggle fieldPath="training.prior_divergence" checked={$projectConfig?.training?.prior_divergence ?? false} onchange={(e) => updateTraining('prior_divergence', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						KL-divergence regularization that penalizes the trained model from diverging too far from the original pretrained model weights.
					</p>
					<FormField type="number" fieldPath="training.prior_divergence_multiplier" value={$projectConfig?.training?.prior_divergence_multiplier ?? 0.1} oninput={(e) => updateTraining('prior_divergence_multiplier', Number(e.target.value))} step="0.01" min={0} tooltip="KL-divergence regularization strength (default 0.1)" />
					<div class="mt-2">
						<FormField fieldPath="training.prior_divergence_args" value={$projectConfig?.training?.prior_divergence_args || ''} oninput={(e) => updateTraining('prior_divergence_args', e.target.value)} placeholder="multiplier=0.1" tooltip="Additional values passed after --prior_divergence_args." />
					</div>
				</div>

				<!-- Precaching options -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Precached Preservation</span>
						<FormToggle fieldPath="training.use_precached_preservation" checked={$projectConfig?.training?.use_precached_preservation ?? false} onchange={(e) => updateTraining('use_precached_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Use pre-cached text encoder outputs for preservation prompts (must be cached during the caching step).
					</p>
					<PathInput fieldPath="training.preservation_prompts_cache" value={$projectConfig?.training?.preservation_prompts_cache || ''} oninput={(e) => updateTraining('preservation_prompts_cache', e.target.value)} showFiles tooltip="Directory with cached preservation embeddings" />
				</div>

				<!-- TARP / DCR -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">TARP (Temporal Aligned RoPE Partitioning)</span>
						<FormToggle fieldPath="training.tarp" checked={$projectConfig?.training?.tarp ?? false} onchange={(e) => updateTraining('tarp', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Windowed cross-attention masks restricting each video frame to temporally nearby audio tokens. Requires AV mode. arXiv:2603.18600.
					</p>
					<FormField type="number" fieldPath="training.tarp_window_multiplier" value={$projectConfig?.training?.tarp_window_multiplier ?? 3} oninput={(e) => updateTraining('tarp_window_multiplier', Number(e.target.value))} step="1" min={1} tooltip="Window size = multiplier * (audio_tokens_per_frame). Default 3." />
					<div class="mt-2">
						<FormField fieldPath="training.tarp_args" value={$projectConfig?.training?.tarp_args || ''} oninput={(e) => updateTraining('tarp_args', e.target.value)} placeholder="window_multiplier=3" tooltip="Additional values passed after --tarp_args." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">DCR (Dynamic Context Routing)</span>
						<FormToggle fieldPath="training.dcr" checked={$projectConfig?.training?.dcr ?? false} onchange={(e) => updateTraining('dcr', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Per-sample gradient detachment in cross-attention for mixed audio/video batches. Detaches absent-audio and clean-reference streams. Requires AV mode. arXiv:2603.18600.
					</p>
					<FormToggle fieldPath="training.dcr_reference_detach" checked={$projectConfig?.training?.dcr_reference_detach ?? true} onchange={(e) => updateTraining('dcr_reference_detach', e.target.checked)} tooltip="Detach gradients when sigma=0 (clean reference conditioning)" />
					<div class="mt-2">
						<FormField fieldPath="training.dcr_args" value={$projectConfig?.training?.dcr_args || ''} oninput={(e) => updateTraining('dcr_args', e.target.value)} placeholder="reference_detach=false" tooltip="Additional values passed after --dcr_args." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">AV Attention Loss</span>
						<FormToggle fieldPath="training.av_attention_loss_weighting" checked={$projectConfig?.training?.av_attention_loss_weighting ?? false} onchange={(e) => updateTraining('av_attention_loss_weighting', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Uses A2V/V2A attention concentration to upweight selected denoising loss tokens. Requires AV mode.
					</p>
					<div class="grid grid-cols-2 gap-2">
						<FormField type="number" fieldPath="training.av_attention_loss_max" value={$projectConfig?.training?.av_attention_loss_max ?? 1.5} oninput={(e) => updateTraining('av_attention_loss_max', Number(e.target.value))} step="0.05" min={1} disabled={!$projectConfig?.training?.av_attention_loss_weighting} tooltip="Maximum token loss multiplier. Default 1.5." />
						<FormField type="number" fieldPath="training.av_attention_loss_warmup_steps" value={$projectConfig?.training?.av_attention_loss_warmup_steps ?? 400} oninput={(e) => updateTraining('av_attention_loss_warmup_steps', Number(e.target.value))} step="50" min={0} disabled={!$projectConfig?.training?.av_attention_loss_weighting} tooltip="Steps before reaching the maximum multiplier." />
					</div>
				</div>

				<!-- Audio Quality Metrics -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Audio Quality Metrics</span>
						<FormToggle fieldPath="training.audio_metrics" checked={$projectConfig?.training?.audio_metrics ?? false} onchange={(e) => updateTraining('audio_metrics', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Latent-space metrics (FD, temporal coherence, AV sync) run every step at ~0 cost. Mel-space and embedding-space metrics are opt-in.
					</p>
					<div class="grid grid-cols-3 gap-x-4 gap-y-1 mb-2">
						<FormToggle fieldPath="training.audio_metrics_mel_metrics" checked={$projectConfig?.training?.audio_metrics_mel_metrics ?? false} onchange={(e) => updateTraining('audio_metrics_mel_metrics', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Spectral convergence, MCD, log-spectral distance (periodic, requires VAE decode)" />
						<FormToggle fieldPath="training.audio_metrics_clap_similarity" checked={$projectConfig?.training?.audio_metrics_clap_similarity ?? false} onchange={(e) => updateTraining('audio_metrics_clap_similarity', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="CLAP audio-text cosine similarity at sampling time" />
						<FormToggle fieldPath="training.audio_metrics_av_onset_alignment" checked={$projectConfig?.training?.audio_metrics_av_onset_alignment ?? false} onchange={(e) => updateTraining('audio_metrics_av_onset_alignment', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Correlation between audio onsets and video motion at sampling time" />
					</div>
					<FormField type="number" fieldPath="training.audio_metrics_mel_compute_every" value={$projectConfig?.training?.audio_metrics_mel_compute_every ?? 100} oninput={(e) => updateTraining('audio_metrics_mel_compute_every', Number(e.target.value))} step="10" min={1} disabled={!$projectConfig?.training?.audio_metrics_mel_metrics} tooltip="Compute mel metrics every N steps" />
					<div class="mt-2">
						<FormField fieldPath="training.audio_metrics_args" value={$projectConfig?.training?.audio_metrics_args || ''} oninput={(e) => updateTraining('audio_metrics_args', e.target.value)} placeholder="key=value ..." tooltip="Additional values passed after --audio_metrics_args." />
					</div>
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
						<div class="text-[11px]" style="color: var(--text-muted);">Train controllable sliders from prompt pairs, paired caches, or IC-aware v2v pairs</div>
					</div>
				</div>
			</div>

			<!-- Config -->
			<div class="p-5 space-y-4">
				<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
					Model, LoRA, optimizer, memory, and output settings are inherited from the Training tab. Slider-specific cache paths and mode selection live here.
				</p>

				<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
					<!-- Left: Slider settings -->
					<div class="space-y-3">
						<FormGroup title="Slider Settings">
							<div class="space-y-2 pt-2">
								<FormSelect
									fieldPath="slider.mode"
									label="Mode"
									value={$projectConfig?.slider?.mode || 'text'}
									options={[
										{ value: 'text', label: 'text' },
										{ value: 'reference', label: 'reference' },
										{ value: 'ic_reference', label: 'ic_reference (v2v)' }
									]}
									onchange={(e) => update('mode', e.target.value)}
									tooltip="Slider training mode. ic_reference currently reuses the v2v IC-LoRA path."
								/>
								<div class="grid grid-cols-2 gap-2">
									<FormField type="number" fieldPath="slider.max_train_steps" value={$projectConfig?.slider?.max_train_steps ?? 500} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Slider training steps (typically less than full training)" />
									<FormField fieldPath="slider.output_name" value={$projectConfig?.slider?.output_name || 'ltx2_slider'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix for slider LoRA" />
								</div>
								{#if ($projectConfig?.slider?.mode || 'text') === 'text'}
									<FormField type="number" fieldPath="slider.guidance_strength" value={$projectConfig?.slider?.guidance_strength ?? 1.0} oninput={(e) => update('guidance_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance strength for text-mode training" />
									<div class="grid grid-cols-3 gap-2">
										<FormField fieldPath="slider.latent_frames" label="Frames" type="number" value={$projectConfig?.slider?.latent_frames ?? 1} oninput={(e) => update('latent_frames', Number(e.target.value))} min={1} tooltip="Latent frames (1=image, >1=video)" />
										<FormField type="number" fieldPath="slider.latent_height" value={$projectConfig?.slider?.latent_height ?? 512} oninput={(e) => update('latent_height', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent height" />
										<FormField type="number" fieldPath="slider.latent_width" value={$projectConfig?.slider?.latent_width ?? 768} oninput={(e) => update('latent_width', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent width" />
									</div>
								{:else}
									<div class="grid grid-cols-2 gap-2">
										<PathInput fieldPath="slider.pos_cache_dir" value={$projectConfig?.slider?.pos_cache_dir || ''} oninput={(e) => update('pos_cache_dir', e.target.value)} showFiles tooltip="Directory with positive latent caches" />
										<PathInput fieldPath="slider.neg_cache_dir" value={$projectConfig?.slider?.neg_cache_dir || ''} oninput={(e) => update('neg_cache_dir', e.target.value)} showFiles tooltip="Directory with negative latent caches" />
									</div>
									<div class="grid grid-cols-2 gap-2">
										<PathInput fieldPath="slider.text_cache_dir" value={$projectConfig?.slider?.text_cache_dir || ''} oninput={(e) => update('text_cache_dir', e.target.value)} showFiles tooltip="Directory with matching text embedding caches" />
										{#if ($projectConfig?.slider?.mode || 'text') === 'reference'}
											<FormSelect fieldPath="slider.reference_modality" value={$projectConfig?.slider?.reference_modality || 'video'} options={['video', 'audio']} onchange={(e) => update('reference_modality', e.target.value)} tooltip="Paired slider target modality" />
										{:else}
											<PathInput fieldPath="slider.reference_cache_dir" value={$projectConfig?.slider?.reference_cache_dir || ''} oninput={(e) => update('reference_cache_dir', e.target.value)} showFiles tooltip="Reference latent cache directory used for the shared v2v IC context" />
										{/if}
									</div>
									<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
										{#if ($projectConfig?.slider?.mode || 'text') === 'ic_reference'}
											`ic_reference` currently implements a shared-reference `v2v` slider: the positive and negative targets use the same cached visual reference clip.
										{:else}
											Reference sliders train from paired cached examples instead of prompt targets. Use audio modality only with `--ltx2_mode audio`.
										{/if}
									</p>
								{/if}
								<FormField fieldPath="slider.sample_slider_range" value={$projectConfig?.slider?.sample_slider_range || '-2,-1,0,1,2'} oninput={(e) => update('sample_slider_range', e.target.value)} tooltip="Comma-separated multiplier values for preview sampling" />
								<FormField fieldPath="slider.accelerate_extra_args" value={$projectConfig?.slider?.accelerate_extra_args || ''} oninput={(e) => update('accelerate_extra_args', e.target.value)} placeholder="--num_processes 2 --main_process_port 29502" tooltip="Extra arguments appended to `accelerate launch` before the slider training script path." />
								<FormField fieldPath="slider.extra_args" value={$projectConfig?.slider?.extra_args || ''} oninput={(e) => update('extra_args', e.target.value)} placeholder="--flag value --other_flag" tooltip="Extra arguments appended to the slider training script command. Use this for any CLI option without a dedicated dashboard control." />
							</div>
						</FormGroup>
					</div>

					<!-- Right: Targets -->
					<div class="space-y-3">
						<FormGroup title="Slider Targets">
							<div class="space-y-3 pt-2">
								{#if ($projectConfig?.slider?.mode || 'text') === 'text'}
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
												<span class="flex items-center gap-1 text-[10px] font-medium mb-0.5" style="color: var(--success);">
													<span>Positive (+)</span>
													<FieldResetButton fieldPath={`slider.targets.${i}.positive`} />
												</span>
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
												<span class="flex items-center gap-1 text-[10px] font-medium mb-0.5" style="color: var(--danger);">
													<span>Negative (-)</span>
													<FieldResetButton fieldPath={`slider.targets.${i}.negative`} />
												</span>
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
												<FormField fieldPath={`slider.targets.${i}.target_class`} label="Target Class" value={target.target_class || ''} oninput={(e) => updateTarget(i, 'target_class', e.target.value)} placeholder="(all content)" tooltip="Optional: restrict to class" />
												<FormField fieldPath={`slider.targets.${i}.weight`} label="Weight" type="number" value={target.weight ?? 1.0} oninput={(e) => updateTarget(i, 'weight', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for this target" />
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
								{:else}
									<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
										Reference-based slider modes use paired cached examples instead of prompt targets. The positive and negative samples must share basename-aligned cache files.
									</p>
								{/if}
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
