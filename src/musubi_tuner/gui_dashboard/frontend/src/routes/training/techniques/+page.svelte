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
			<p class="text-[12px]" style="color: var(--text-muted);">Optional training objectives, routing methods, and preservation tools. If you are unsure, leave these off and enable one technique at a time so changes are easy to diagnose.</p>
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
					Aligns intermediate DiT representations across video frames during fine-tuning, improving temporal consistency. T2V default: schedule=cosine, lambda=0.5, warmup=100, decay steps=0 (full run), lambda end=0.1, threshold=0.85, cutoff=permanent. I2V suggestion: schedule=constant, lambda=0.5; the reference frame usually anchors consistency.
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
						<FormField fieldPath="training.crepa_lambda" label="Lambda" type="number" value={$projectConfig?.training?.crepa_lambda ?? 0.5} oninput={(e) => updateTraining('crepa_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="CREPA starting loss weight. T2V and I2V suggestion: 0.5." />
						<FormField fieldPath="training.crepa_lambda_end" label="Lambda End" type="number" value={$projectConfig?.training?.crepa_lambda_end ?? 0.1} oninput={(e) => updateTraining('crepa_lambda_end', Number(e.target.value))} step="0.01" min={0} tooltip="Ending CREPA weight for linear/cosine schedules. T2V suggestion: 0.1. Constant schedule ignores this." />
						<FormField fieldPath="training.crepa_tau" label="Tau" type="number" value={$projectConfig?.training?.crepa_tau ?? 1.0} oninput={(e) => updateTraining('crepa_tau', Number(e.target.value))} step="0.1" min={0.01} tooltip="Temporal neighbor decay factor (default 1.0)" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField fieldPath="training.crepa_num_neighbors" label="Neighbors" type="number" value={$projectConfig?.training?.crepa_num_neighbors ?? 2} oninput={(e) => updateTraining('crepa_num_neighbors', Number(e.target.value))} min={1} max={8} tooltip="K frames on each side for alignment (default 2)" />
						<FormSelect fieldPath="training.crepa_schedule" label="Schedule" value={$projectConfig?.training?.crepa_schedule || 'cosine'} onchange={(e) => updateTraining('crepa_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="T2V default: cosine decay from Lambda to Lambda End. I2V suggestion: constant." />
						<FormField fieldPath="training.crepa_warmup_steps" label="Warmup Steps" type="number" value={$projectConfig?.training?.crepa_warmup_steps ?? 100} oninput={(e) => updateTraining('crepa_warmup_steps', Number(e.target.value))} min={0} tooltip="Steps before CREPA reaches full starting strength. T2V default: 100. I2V suggestion: 0." />
					</div>
					<div class="grid grid-cols-3 gap-2 mt-2">
						<FormField fieldPath="training.crepa_decay_steps" label="Decay Steps" type="number" value={$projectConfig?.training?.crepa_decay_steps ?? 0} oninput={(e) => updateTraining('crepa_decay_steps', Number(e.target.value))} min={0} tooltip="Steps used for linear/cosine decay after warmup. 0 means auto-use max training steps." />
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.crepa_normalize" checked={$projectConfig?.training?.crepa_normalize ?? true} onchange={(e) => updateTraining('crepa_normalize', e.target.checked)} tooltip="L2-normalize features before similarity computation" />
						</div>
					</div>
				</div>
				<FormField fieldPath="training.crepa_args" value={$projectConfig?.training?.crepa_args || ''} oninput={(e) => updateTraining('crepa_args', e.target.value)} placeholder="key=value ..." tooltip="Additional values passed after --crepa_args." />

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Similarity Cutoff</div>
					<p class="text-[10px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Cutoff is independent of the CREPA weight schedule. It stops CREPA once the EMA alignment score reaches the threshold. T2V default: threshold=0.85, EMA=0.99, mode=permanent. If you only want similarity-based cutoff, these three fields are the required ones.
					</p>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Threshold" type="number" value={$projectConfig?.training?.crepa_similarity_threshold ?? ''} oninput={(e) => updateTraining('crepa_similarity_threshold', e.target.value ? Number(e.target.value) : null)} placeholder="Off" step="0.01" min={0} max={0.99} tooltip="EMA alignment threshold. Blank disables similarity cutoff." />
						<FormField label="EMA Decay" type="number" value={$projectConfig?.training?.crepa_similarity_ema_decay ?? 0.99} oninput={(e) => updateTraining('crepa_similarity_ema_decay', Number(e.target.value))} step="0.005" min={0} max={0.999} tooltip="Smoothing factor for the alignment score EMA." />
						<FormSelect label="Mode" value={$projectConfig?.training?.crepa_threshold_mode || 'permanent'} onchange={(e) => updateTraining('crepa_threshold_mode', e.target.value)} options={[{value: 'permanent', label: 'Permanent'}, {value: 'recoverable', label: 'Recoverable'}]} tooltip="Permanent keeps CREPA off after cutoff. Recoverable can re-enable if the EMA drops below threshold." />
					</div>
				</div>

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
						<div class="text-[11px]" style="color: var(--text-muted);">Sparse video-token routing during selected transformer layers</div>
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Token routing skips or routes a subset of video tokens through selected transformer layers to reduce training cost. Safe default: off. Blank layer fields use SimpleTuner's LTX-2 default route, layers 2 through -2.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-2">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">TREAD</span>
						<FormToggle checked={$projectConfig?.training?.tread ?? false} onchange={(e) => updateTraining('tread', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Simpler sparse routing. Default selection ratio 0.5 means roughly half the tokens are routed through the selected layer window. Increase only if quality holds; lower if motion/detail degrades.
					</p>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Selection Ratio" type="number" value={$projectConfig?.training?.tread_selection_ratio ?? 0.5} oninput={(e) => updateTraining('tread_selection_ratio', Number(e.target.value))} step="0.05" min={0} max={0.95} disabled={!$projectConfig?.training?.tread} tooltip="Fraction of tokens affected by TREAD routing. Default 0.5. Higher saves more compute but can hurt detail or temporal consistency." />
						<FormField label="Start Layer" type="number" value={$projectConfig?.training?.tread_start_layer_idx ?? ''} oninput={(e) => updateTraining('tread_start_layer_idx', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!$projectConfig?.training?.tread} tooltip="First transformer layer to route. Blank uses trainer default: 3 for LTX-2.3, 2 for LTX-2.0. Early layers are more sensitive." />
						<FormField label="End Layer" type="number" value={$projectConfig?.training?.tread_end_layer_idx ?? ''} oninput={(e) => updateTraining('tread_end_layer_idx', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" disabled={!$projectConfig?.training?.tread} tooltip="Last routed layer. Negative values count from the end. Blank uses trainer default: -4 for LTX-2.3, -2 for LTX-2.0." />
					</div>
				</div>
			</div>
		</div>

		<!-- Differential Guidance -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M3.75 13.5l6-6m0 0v4.5m0-4.5H5.25m15 3l-6 6m0 0v-4.5m0 4.5h4.5"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Differential Guidance</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Prediction-relative target scaling for the main training loss</div>
					</div>
					<div class="ml-auto">
						<FormToggle fieldPath="training.differential_guidance" checked={$projectConfig?.training?.differential_guidance ?? false} onchange={(e) => updateTraining('differential_guidance', e.target.checked)} />
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Amplifies or softens the difference between the current prediction and the training target, matching ai-toolkit's formula: target = pred + scale * (target - pred). This can make a weak concept or motion signal learn faster, but it can also over-steer training, increase artifacts, or destabilize small datasets. Default scale 3 strengthens the difference; values between 0 and 1, such as 0.5, soften it; negative values push the opposite direction.
				</p>
			</div>

			<div class="p-5 pt-0">
				<div class="grid grid-cols-4 gap-2">
					<FormField fieldPath="training.differential_guidance_scale" label="Scale" type="number" value={$projectConfig?.training?.differential_guidance_scale ?? 3.0} oninput={(e) => updateTraining('differential_guidance_scale', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.differential_guidance} tooltip="Multiplier for target = pred + scale * (target - pred). Default 3.0. 0.5 softens the target difference; negative values push the opposite direction." />
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
					Self-Flow adds a feature-alignment regularizer between a student block and a teacher signal at different noise levels. Safe default: off. If you enable it, start with Teacher Mode = Base, Lambda = 0.1, Mask Ratio = 0.1, Dual Timestep on, and leave temporal consistency off until the base Self-Flow loss is stable.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Architecture</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Chooses where features are captured and what acts as the teacher. Ratio fields are the recommended way to pick blocks across model depths; they override the explicit block index fields.
					</p>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_teacher_mode" label="Teacher Mode" value={$projectConfig?.training?.self_flow_teacher_mode ?? 'base'} onchange={(e) => updateTraining('self_flow_teacher_mode', e.target.value)} options={[{value: 'base', label: 'Base model'}, {value: 'ema', label: 'EMA (all LoRA)'}, {value: 'partial_ema', label: 'Partial EMA (teacher block)'}]} tooltip="Base is the safest default and uses a frozen pretrained teacher. EMA tracks the LoRA during training and can add memory/state overhead. Partial EMA only tracks the teacher block's LoRA params." />
						<FormField fieldPath="training.self_flow_student_block_idx" label="Student Block" type="number" value={$projectConfig?.training?.self_flow_student_block_idx ?? 16} oninput={(e) => updateTraining('self_flow_student_block_idx', Number(e.target.value))} min={0} max={47} tooltip="Explicit student feature block index. Default 16. Ignored when Student Ratio is set, which is the normal GUI default." />
						<FormField fieldPath="training.self_flow_teacher_block_idx" label="Teacher Block" type="number" value={$projectConfig?.training?.self_flow_teacher_block_idx ?? 32} oninput={(e) => updateTraining('self_flow_teacher_block_idx', Number(e.target.value))} min={0} max={47} tooltip="Explicit teacher feature block index. Default 32 and should be deeper than student. Ignored when Teacher Ratio is set." />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField type="number" fieldPath="training.self_flow_student_block_ratio" value={$projectConfig?.training?.self_flow_student_block_ratio ?? 0.3} oninput={(e) => updateTraining('self_flow_student_block_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Ratio-based student block: floor(ratio × depth). Takes priority over block index." />
						<FormField type="number" fieldPath="training.self_flow_teacher_block_ratio" value={$projectConfig?.training?.self_flow_teacher_block_ratio ?? 0.7} oninput={(e) => updateTraining('self_flow_teacher_block_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Ratio-based teacher block: ceil(ratio × depth). Takes priority over block index." />
						<FormField type="number" fieldPath="training.self_flow_student_block_stochastic_range" value={$projectConfig?.training?.self_flow_student_block_stochastic_range ?? 0} oninput={(e) => updateTraining('self_flow_student_block_stochastic_range', Number(e.target.value))} min={0} max={8} tooltip="Randomly vary student capture block ±N each step. 0 = fixed block. Adds regularization variety but uses a shared projector across depths." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Parameters</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Controls the strength of Self-Flow and the dual-timestep mask. Keep Lambda and Mask Ratio small at first; use Max Loss Cap if the Self-Flow loss overwhelms the main denoising loss.
					</p>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField fieldPath="training.self_flow_lambda" label="Lambda" type="number" value={$projectConfig?.training?.self_flow_lambda ?? 0.1} oninput={(e) => updateTraining('self_flow_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="Base Self-Flow loss weight. Default 0.1. Reduce if it hurts prompt/content learning; increase carefully if the regularizer is too weak." />
						<FormField fieldPath="training.self_flow_mask_ratio" label="Mask Ratio" type="number" value={$projectConfig?.training?.self_flow_mask_ratio ?? 0.1} oninput={(e) => updateTraining('self_flow_mask_ratio', Number(e.target.value))} step="0.05" min={0} max={0.5} tooltip="Fraction of tokens or frames that receive the alternate timestep. Default 0.1. Valid range 0 to 0.5; higher is stronger and riskier." />
						<FormField fieldPath="training.self_flow_max_loss" label="Max Loss Cap" type="number" value={$projectConfig?.training?.self_flow_max_loss ?? 0.0} oninput={(e) => updateTraining('self_flow_max_loss', Number(e.target.value))} step="0.01" min={0} placeholder="Disabled" tooltip="Rescales Self-Flow if its magnitude exceeds this cap. Default 0 disabled. Use only if Self-Flow dominates early training." />
					</div>
					<div class="grid grid-cols-4 gap-2 mb-2">
						<FormField fieldPath="training.self_flow_teacher_momentum" label="Momentum" type="number" value={$projectConfig?.training?.self_flow_teacher_momentum ?? 0.999} oninput={(e) => updateTraining('self_flow_teacher_momentum', Number(e.target.value))} step="0.001" min={0} max={1} tooltip="EMA momentum for teacher updates. Default 0.999. Only used by EMA and Partial EMA teacher modes." />
						<FormField fieldPath="training.self_flow_projector_lr" label="Projector LR" type="number" value={$projectConfig?.training?.self_flow_projector_lr ?? ''} oninput={(e) => updateTraining('self_flow_projector_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Same as LR" step="any" tooltip="Optional separate learning rate for the Self-Flow projector MLP. Blank uses the main training LR." />
						<FormSelect fieldPath="training.self_flow_projector_activation" label="Projector Act" value={$projectConfig?.training?.self_flow_projector_activation ?? 'silu'} onchange={(e) => updateTraining('self_flow_projector_activation', e.target.value)} options={['silu', 'gelu']} tooltip="Activation in the Self-Flow projector. Default SiLU. GELU is available for experiments." />
						<FormField fieldPath="training.self_flow_lambda_audio" label="Audio Lambda" type="number" value={$projectConfig?.training?.self_flow_lambda_audio ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_audio', Number(e.target.value))} step="0.01" min={0} tooltip="Optional Self-Flow audio feature loss weight for AV mode. Default 0 off. Start very small if testing." />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_dual_timestep" label="Dual Timestep" checked={$projectConfig?.training?.self_flow_dual_timestep ?? true} onchange={(e) => updateTraining('self_flow_dual_timestep', e.target.checked)} tooltip="Default on. Applies the alternate timestep mask that Self-Flow is built around. Turn off only for ablations." />
						</div>
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_frame_level_mask" label="Frame-Level Mask" checked={$projectConfig?.training?.self_flow_frame_level_mask ?? false} onchange={(e) => updateTraining('self_flow_frame_level_mask', e.target.checked)} tooltip="Masks whole frames instead of individual tokens. Default off. More semantically meaningful for video but stronger than token masking." />
						</div>
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_mask_focus_loss" label="Mask-Focus Loss" checked={$projectConfig?.training?.self_flow_mask_focus_loss ?? false} onchange={(e) => updateTraining('self_flow_mask_focus_loss', e.target.checked)} tooltip="Computes representation loss only on masked higher-noise tokens. Default off uses all tokens, which is usually more stable." />
						</div>
					</div>
					<div class="grid grid-cols-1 gap-2 mt-2">
						<div class="flex items-end pb-0.5">
							<FormToggle fieldPath="training.self_flow_offload_teacher_features" label="Offload Teacher Features" checked={$projectConfig?.training?.self_flow_offload_teacher_features ?? false} onchange={(e) => updateTraining('self_flow_offload_teacher_features', e.target.checked)} tooltip="Moves cached teacher features to CPU to reduce VRAM. Default off. Can reduce memory at the cost of CPU transfer overhead." />
						</div>
					</div>
				</div>

				<!-- Temporal Consistency -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-1" style="color: var(--text-primary);">Temporal Consistency</div>
					<div class="text-[11px] mb-2" style="color: var(--text-muted);">Optional add-on losses for frame-neighbor and motion-delta consistency. Safe default: Mode Off and both lambdas 0. Enable only after base Self-Flow is stable.</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_temporal_mode" label="Mode" value={$projectConfig?.training?.self_flow_temporal_mode ?? 'off'} onchange={(e) => updateTraining('self_flow_temporal_mode', e.target.value)} options={[{value: 'off', label: 'Off'}, {value: 'frame', label: 'Frame'}, {value: 'delta', label: 'Delta'}, {value: 'hybrid', label: 'Hybrid'}]} tooltip="Off is default. Frame aligns neighboring frames. Delta aligns motion differences. Hybrid applies both and is the strongest." />
						<FormField fieldPath="training.self_flow_lambda_temporal" label="Lambda Temporal" type="number" value={$projectConfig?.training?.self_flow_lambda_temporal ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_temporal', Number(e.target.value))} step="0.01" min={0} tooltip="Frame-neighbor alignment weight for frame/hybrid modes. Default 0. Start tiny, for example 0.01." />
						<FormField fieldPath="training.self_flow_lambda_delta" label="Lambda Delta" type="number" value={$projectConfig?.training?.self_flow_lambda_delta ?? 0.0} oninput={(e) => updateTraining('self_flow_lambda_delta', Number(e.target.value))} step="0.01" min={0} tooltip="Motion-delta alignment weight for delta/hybrid modes. Default 0. Start tiny, for example 0.01." />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField fieldPath="training.self_flow_num_neighbors" label="Neighbors" type="number" value={$projectConfig?.training?.self_flow_num_neighbors ?? 2} oninput={(e) => updateTraining('self_flow_num_neighbors', Number(e.target.value))} min={0} max={8} tooltip="Temporal neighbors on each side for frame alignment. Default 2. Higher looks farther across time and costs more." />
						<FormField fieldPath="training.self_flow_temporal_tau" label="Tau" type="number" value={$projectConfig?.training?.self_flow_temporal_tau ?? 1.0} oninput={(e) => updateTraining('self_flow_temporal_tau', Number(e.target.value))} step="0.1" min={0.1} tooltip="Neighbor weight decay factor. Default 1.0. Higher decays slower, giving farther neighbors more influence." />
						<FormField fieldPath="training.self_flow_delta_num_steps" label="Delta Steps" type="number" value={$projectConfig?.training?.self_flow_delta_num_steps ?? 1} oninput={(e) => updateTraining('self_flow_delta_num_steps', Number(e.target.value))} min={1} max={8} tooltip="Motion delta span. Default 1 compares adjacent frames only; higher compares longer motion differences." />
					</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_temporal_granularity" label="Granularity" value={$projectConfig?.training?.self_flow_temporal_granularity ?? 'frame'} onchange={(e) => updateTraining('self_flow_temporal_granularity', e.target.value)} options={[{value: 'frame', label: 'Frame'}, {value: 'patch', label: 'Patch'}]} tooltip="Frame is default and faster: mean-pooled per frame. Patch is spatial per-token matching, stronger but more expensive." />
						<FormField fieldPath="training.self_flow_patch_spatial_radius" label="Patch Radius" type="number" value={$projectConfig?.training?.self_flow_patch_spatial_radius ?? 0} oninput={(e) => updateTraining('self_flow_patch_spatial_radius', Number(e.target.value))} min={0} max={4} tooltip="Local spatial search radius for patch matching. Default 0 requires same position. Only used with patch granularity." />
						<FormSelect fieldPath="training.self_flow_patch_match_mode" label="Patch Mode" value={$projectConfig?.training?.self_flow_patch_match_mode ?? 'hard'} onchange={(e) => updateTraining('self_flow_patch_match_mode', e.target.value)} options={[{value: 'hard', label: 'Hard'}, {value: 'soft', label: 'Soft'}]} tooltip="Hard picks the best local patch match. Soft uses softmax-weighted matches. Hard is the default." />
					</div>
					<div class="grid grid-cols-2 gap-2 mb-2">
						<FormSelect fieldPath="training.self_flow_motion_weighting" label="Motion Weighting" value={$projectConfig?.training?.self_flow_motion_weighting ?? 'none'} onchange={(e) => updateTraining('self_flow_motion_weighting', e.target.value)} options={[{value: 'none', label: 'None'}, {value: 'teacher_delta', label: 'Teacher Delta'}]} tooltip="Default None. Teacher Delta upweights regions where teacher features show more motion." />
						<FormField fieldPath="training.self_flow_motion_weight_strength" label="Motion Strength" type="number" value={$projectConfig?.training?.self_flow_motion_weight_strength ?? 0.0} oninput={(e) => updateTraining('self_flow_motion_weight_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Strength of motion-based weighting. Default 0.0. Only matters when Motion Weighting is enabled." />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormSelect fieldPath="training.self_flow_temporal_schedule" label="Schedule" value={$projectConfig?.training?.self_flow_temporal_schedule ?? 'constant'} onchange={(e) => updateTraining('self_flow_temporal_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="Schedule for all Self-Flow lambdas. Default constant. Linear/cosine decay toward zero by Max Steps." />
						<FormField fieldPath="training.self_flow_temporal_warmup_steps" label="Warmup Steps" type="number" value={$projectConfig?.training?.self_flow_temporal_warmup_steps ?? 0} oninput={(e) => updateTraining('self_flow_temporal_warmup_steps', Number(e.target.value))} min={0} tooltip="Linear ramp-up before Self-Flow/temporal losses reach full weight. Default 0." />
						<FormField fieldPath="training.self_flow_temporal_max_steps" label="Max Steps" type="number" value={$projectConfig?.training?.self_flow_temporal_max_steps ?? 0} oninput={(e) => updateTraining('self_flow_temporal_max_steps', Number(e.target.value))} min={0} tooltip="Step where linear/cosine schedule reaches zero. Default 0 disables decay." />
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
					Degrades clean latents via downsample-upsample before noise addition, then trains the model to reconstruct the original clean latents. Safe default: off. Use for image-only or low-motion data when you want the LoRA to retain high-frequency detail without sacrificing video temporal behavior; the existing defaults are the recommended starting point.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Parameters</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField fieldPath="training.hfato_scale_factor" label="Scale Factor" type="number" value={$projectConfig?.training?.hfato_scale_factor ?? 0.5} oninput={(e) => updateTraining('hfato_scale_factor', Number(e.target.value))} step="0.05" min={0.05} max={1.0} tooltip="Downsample ratio for spatial degradation. Default 0.5 halves each spatial dimension. Lower is more aggressive and can make reconstruction harder." />
						<FormSelect fieldPath="training.hfato_interpolation" label="Interpolation" value={$projectConfig?.training?.hfato_interpolation || 'bilinear'} onchange={(e) => updateTraining('hfato_interpolation', e.target.value)} options={[{value: 'bilinear', label: 'Bilinear'}, {value: 'nearest', label: 'Nearest'}, {value: 'bicubic', label: 'Bicubic'}]} tooltip="Resampling method for the downsample-upsample degradation. Default bilinear is the balanced option." />
						<FormField fieldPath="training.hfato_probability" label="Probability" type="number" value={$projectConfig?.training?.hfato_probability ?? 1.0} oninput={(e) => updateTraining('hfato_probability', Number(e.target.value))} step="0.05" min={0.05} max={1.0} tooltip="Chance to apply HFATO on a training step. Default 1.0 always applies it. Lower if the auxiliary objective is too strong." />
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
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					These controls only affect audio or audio-video LTX-2 modes. Safe default: leave the experimental loss balancers and regularizers off unless your run is clearly under-training audio or mixing audio/no-audio samples. For first AV runs, Audio Supervision = Warn is useful to catch missing audio latents before wasting a run.
				</p>
			</div>

			<div class="p-5 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Balance</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Controls how video loss and audio loss are combined. Default None keeps the trainer's static weights. Try Inverse Frequency when audio batches are rare; try EMA Magnitude when the audio loss scale is much larger or smaller than video. Uncertainty and OGM-GE are more experimental.
					</p>
					<FormSelect fieldPath="training.audio_loss_balance_mode" label="Mode" value={$projectConfig?.training?.audio_loss_balance_mode || 'none'} onchange={(e) => updateTraining('audio_loss_balance_mode', e.target.value)} options={[{value: 'none', label: 'None (static weights)'}, {value: 'inv_freq', label: 'Inverse Frequency'}, {value: 'ema_mag', label: 'EMA Magnitude'}, {value: 'uncertainty', label: 'Uncertainty'}, {value: 'ogm_ge', label: 'OGM-GE'}]} tooltip="None: static weights, safest default. Inverse Frequency: boosts audio when audio batches are sparse. EMA Magnitude: adjusts audio weight toward a target audio/video loss ratio. Uncertainty: learns video/audio log-variance weights. OGM-GE: attenuates the faster/lower-loss modality each step." />
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField fieldPath="training.audio_loss_balance_ema_init" label="EMA Init" type="number" value={$projectConfig?.training?.audio_loss_balance_ema_init ?? 1.0} oninput={(e) => updateTraining('audio_loss_balance_ema_init', Number(e.target.value))} step="0.1" min={0} tooltip="Initial running value for loss/presence EMAs. Default 1.0 avoids an early weight spike before the trainer has measured real audio/video loss." />
						{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'uncertainty'}
							<FormField fieldPath="training.uncertainty_lr" label="Uncertainty LR" type="number" value={$projectConfig?.training?.uncertainty_lr ?? ''} oninput={(e) => updateTraining('uncertainty_lr', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="any" tooltip="LR for the two learned log-variance scalars. Blank uses the main LR. Lower it if uncertainty weights swing too quickly." />
						{:else if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ogm_ge'}
							<FormField fieldPath="training.ogm_ge_alpha" label="OGM-GE Alpha" type="number" value={$projectConfig?.training?.ogm_ge_alpha ?? 0.3} oninput={(e) => updateTraining('ogm_ge_alpha', Number(e.target.value))} step="0.05" min={0} tooltip="Strength of OGM-GE's per-step reweighting. Default 0.3. Higher reacts harder to video/audio imbalance but can make training less steady." />
						{/if}
					</div>
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'inv_freq'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField fieldPath="training.audio_loss_balance_beta" label="Beta" type="number" value={$projectConfig?.training?.audio_loss_balance_beta ?? 0.01} oninput={(e) => updateTraining('audio_loss_balance_beta', Number(e.target.value))} step="0.005" tooltip="EMA update rate for audio-batch frequency. Default 0.01 is slow and stable; higher adapts faster but changes weights more abruptly." />
						<FormField fieldPath="training.audio_loss_balance_eps" label="Eps" type="number" value={$projectConfig?.training?.audio_loss_balance_eps ?? 0.05} oninput={(e) => updateTraining('audio_loss_balance_eps', Number(e.target.value))} step="0.01" tooltip="Minimum denominator for inverse-frequency weighting. Default 0.05 caps the boost when audio batches are very rare." />
					</div>
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField fieldPath="training.audio_loss_balance_min" label="Min Weight" type="number" value={$projectConfig?.training?.audio_loss_balance_min ?? 0.05} oninput={(e) => updateTraining('audio_loss_balance_min', Number(e.target.value))} step="0.01" tooltip="Lower clamp for the dynamic audio loss weight. Default 0.05 prevents the audio branch from being fully ignored." />
						<FormField fieldPath="training.audio_loss_balance_max" label="Max Weight" type="number" value={$projectConfig?.training?.audio_loss_balance_max ?? 4.0} oninput={(e) => updateTraining('audio_loss_balance_max', Number(e.target.value))} step="0.5" tooltip="Upper clamp for the dynamic audio loss weight. Default 4.0 prevents rare audio batches from dominating the whole update." />
					</div>
					{/if}
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ema_mag'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField fieldPath="training.audio_loss_balance_target_ratio" label="Target Ratio" type="number" value={$projectConfig?.training?.audio_loss_balance_target_ratio ?? 0.33} oninput={(e) => updateTraining('audio_loss_balance_target_ratio', Number(e.target.value))} step="0.05" tooltip="Target audio loss magnitude relative to video loss. Default 0.33 keeps audio important without letting it dominate." />
						<FormField fieldPath="training.audio_loss_balance_ema_decay" label="EMA Decay" type="number" value={$projectConfig?.training?.audio_loss_balance_ema_decay ?? 0.99} oninput={(e) => updateTraining('audio_loss_balance_ema_decay', Number(e.target.value))} step="0.005" tooltip="Smoothing for measured audio/video loss magnitude. Default 0.99 is stable; lower values react faster." />
					</div>
					{/if}
					{#if ($projectConfig?.training?.audio_loss_balance_mode || 'none') === 'ogm_ge'}
					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField fieldPath="training.ogm_ge_noise_std" label="OGM-GE Noise" type="number" value={$projectConfig?.training?.ogm_ge_noise_std ?? 0.0} oninput={(e) => updateTraining('ogm_ge_noise_std', Number(e.target.value))} step="0.01" min={0} tooltip="Optional gradient noise scale for OGM-GE. Default 0.0. Leave off unless you are deliberately testing OGM-GE regularization." />
						<div></div>
					</div>
					{/if}
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Options</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Extra audio training behavior. Defaults are off because these change the training objective. Use Silence Regularizer for mixed datasets with missing audio; use Audio DOP when you need to preserve the base model's audio behavior.
					</p>
					<div class="grid grid-cols-3 gap-x-4 gap-y-1 mb-2">
						<FormToggle fieldPath="training.independent_audio_timestep" label="Independent Timestep" checked={$projectConfig?.training?.independent_audio_timestep ?? false} onchange={(e) => updateTraining('independent_audio_timestep', e.target.checked)} tooltip="Samples a separate noise timestep for audio instead of sharing the video timestep. Default off. Can improve AV robustness but changes the noise distribution." />
						<FormToggle fieldPath="training.audio_silence_regularizer" label="Silence Regularizer" checked={$projectConfig?.training?.audio_silence_regularizer ?? false} onchange={(e) => updateTraining('audio_silence_regularizer', e.target.checked)} tooltip="Adds a synthetic silence target for samples without usable audio. Default off. Useful for mixed AV/no-audio datasets so missing audio does not become random audio." />
						<FormToggle fieldPath="training.audio_dop" label="Audio DOP" checked={$projectConfig?.training?.audio_dop ?? false} onchange={(e) => updateTraining('audio_dop', e.target.checked)} tooltip="Differential Output Preservation for audio: penalizes drift from the base model's audio predictions. Default off. Requires AV mode and costs extra compute." />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField fieldPath="training.audio_silence_regularizer_weight" label="Silence Weight" type="number" value={$projectConfig?.training?.audio_silence_regularizer_weight ?? 1.0} oninput={(e) => updateTraining('audio_silence_regularizer_weight', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.audio_silence_regularizer} tooltip="Loss weight for the synthetic silence objective. Default 1.0. Lower it if silence starts suppressing real audio learning." />
						<FormField fieldPath="training.audio_dop_multiplier" label="DOP Multiplier" type="number" value={$projectConfig?.training?.audio_dop_multiplier ?? 0.5} oninput={(e) => updateTraining('audio_dop_multiplier', Number(e.target.value))} step="0.1" disabled={!$projectConfig?.training?.audio_dop} tooltip="Strength of audio output preservation. GUI default 0.5 is conservative; increase only if audio quality drifts from the base model." />
					</div>
					<div class="mt-2">
						<FormField fieldPath="training.audio_dop_args" value={$projectConfig?.training?.audio_dop_args || ''} oninput={(e) => updateTraining('audio_dop_args', e.target.value)} placeholder="multiplier=0.5" disabled={!$projectConfig?.training?.audio_dop} tooltip="Additional values passed after --audio_dop_args." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Supervision</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Checks whether batches that should have audio actually produce supervised audio loss. This does not improve quality by itself; it catches bad caches or dataset entries. Recommended for new AV datasets: Warn.
					</p>
					<FormSelect fieldPath="training.audio_supervision_mode" label="Mode" value={$projectConfig?.training?.audio_supervision_mode || 'off'} onchange={(e) => updateTraining('audio_supervision_mode', e.target.value)} options={[{value: 'off', label: 'Off'}, {value: 'warn', label: 'Warn'}, {value: 'error', label: 'Error'}]} tooltip="Off: no monitoring. Warn: log an alert if too many expected-audio batches have no audio loss. Error: stop training on that condition. Recommended for first AV test: Warn." />
					{#if ($projectConfig?.training?.audio_supervision_mode || 'off') !== 'off'}
					<div class="grid grid-cols-3 gap-2 mt-2">
						<FormField fieldPath="training.audio_supervision_warmup_steps" label="Warmup" type="number" value={$projectConfig?.training?.audio_supervision_warmup_steps ?? 50} oninput={(e) => updateTraining('audio_supervision_warmup_steps', Number(e.target.value))} min={0} tooltip="Number of expected-audio batches to observe before checking. Default 50 avoids false alarms at startup." />
						<FormField fieldPath="training.audio_supervision_check_interval" label="Interval" type="number" value={$projectConfig?.training?.audio_supervision_check_interval ?? 50} oninput={(e) => updateTraining('audio_supervision_check_interval', Number(e.target.value))} min={1} tooltip="Check every N expected-audio batches after warmup. Default 50." />
						<FormField fieldPath="training.audio_supervision_min_ratio" label="Min Ratio" type="number" value={$projectConfig?.training?.audio_supervision_min_ratio ?? 0.9} oninput={(e) => updateTraining('audio_supervision_min_ratio', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Minimum supervised-audio ratio before warning/error. Default 0.9 means at least 90% of expected-audio batches should have audio loss." />
					</div>
					{/if}
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Audio Buckets & Batch Sampling</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Controls how variable-length audio clips are bucketed and how often audio batches are selected. Leave Auto/Random unless you are tuning mixed audio/video throughput or preventing audio from being starved by video-only batches.
					</p>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormSelect fieldPath="training.audio_bucket_strategy" label="Bucket Strategy" value={$projectConfig?.training?.audio_bucket_strategy || ''} options={[{value:'',label:'Default'},{value:'pad',label:'Pad'},{value:'truncate',label:'Truncate'}]} onchange={(e) => updateTraining('audio_bucket_strategy', e.target.value || null)} tooltip="How to fit audio to bucket length. Default uses trainer behavior, normally pad. Pad keeps the full clip with silence padding. Truncate cuts extra duration." />
						<FormField fieldPath="training.audio_bucket_interval" label="Bucket Interval" type="number" value={$projectConfig?.training?.audio_bucket_interval ?? ''} oninput={(e) => updateTraining('audio_bucket_interval', e.target.value ? Number(e.target.value) : null)} placeholder="Auto" step="0.1" tooltip="Audio duration bucket step in seconds. Auto uses the dataset/trainer default, usually 2.0s. Smaller intervals reduce padding but create more buckets." />
						<FormField fieldPath="training.audio_only_sequence_resolution" label="Audio Seq Res" type="number" value={$projectConfig?.training?.audio_only_sequence_resolution ?? 64} oninput={(e) => updateTraining('audio_only_sequence_resolution', Number(e.target.value))} min={1} tooltip="Sequence resolution for audio-only mode. Default 64. Only change if you know your audio-only latent shape requires it." />
					</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField fieldPath="training.min_audio_batches_per_accum" label="Min Audio Batches" type="number" value={$projectConfig?.training?.min_audio_batches_per_accum ?? 0} oninput={(e) => updateTraining('min_audio_batches_per_accum', Number(e.target.value))} min={0} tooltip="Minimum audio-containing batches per gradient accumulation window. Default 0 disables forcing. Use this if random sampling rarely includes audio." />
						<FormField fieldPath="training.audio_batch_probability" label="Audio Batch Prob" type="number" value={$projectConfig?.training?.audio_batch_probability ?? ''} oninput={(e) => updateTraining('audio_batch_probability', e.target.value ? Number(e.target.value) : null)} placeholder="Random" step="0.1" min={0} max={1} tooltip="Probability of selecting an audio batch when the sampler can choose. Blank means trainer default/random. Higher values bias training toward audio." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Cross-Token Sync & Modality Freeze</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Advanced AV coupling tools. CTS adds an explicit sync loss between audio and video tokens. Modality Freeze temporarily freezes the modality that appears to be converging faster so the weaker branch can catch up. Safe defaults are all off.
					</p>
					<div class="grid grid-cols-2 gap-2 mb-2">
						<FormField fieldPath="training.cts_lambda_video_driven" label="CTS Video" type="number" value={$projectConfig?.training?.cts_lambda_video_driven ?? 0.0} oninput={(e) => updateTraining('cts_lambda_video_driven', Number(e.target.value))} step="0.01" min={0} tooltip="Cross-token sync loss weight using video as the driver. Default 0 off. Start tiny, for example 0.01, if AV sync is weak." />
						<FormField fieldPath="training.cts_lambda_audio_driven" label="CTS Audio" type="number" value={$projectConfig?.training?.cts_lambda_audio_driven ?? 0.0} oninput={(e) => updateTraining('cts_lambda_audio_driven', Number(e.target.value))} step="0.01" min={0} tooltip="Cross-token sync loss weight using audio as the driver. Default 0 off. Start tiny, for example 0.01, if audio should strongly drive motion." />
					</div>
					<div class="grid grid-cols-4 gap-2">
						<FormField fieldPath="training.modality_freeze_check_interval" label="Freeze Interval" type="number" value={$projectConfig?.training?.modality_freeze_check_interval ?? 0} oninput={(e) => updateTraining('modality_freeze_check_interval', Number(e.target.value))} min={0} tooltip="Check interval in steps for automatic modality freezing. Default 0 disables it. Try large intervals such as 500+ if testing." />
						<FormField fieldPath="training.modality_freeze_ratio_threshold" label="Ratio Threshold" type="number" value={$projectConfig?.training?.modality_freeze_ratio_threshold ?? 0.5} oninput={(e) => updateTraining('modality_freeze_ratio_threshold', Number(e.target.value))} step="0.05" min={0} tooltip="Loss-ratio threshold for deciding one modality is ahead. Default 0.5 means freeze only when the audio/video loss ratio is outside roughly 0.5x to 2x." />
						<FormField fieldPath="training.modality_freeze_warmup_steps" label="Warmup" type="number" value={$projectConfig?.training?.modality_freeze_warmup_steps ?? 100} oninput={(e) => updateTraining('modality_freeze_warmup_steps', Number(e.target.value))} min={0} tooltip="Steps before freeze decisions can happen. Default 100 gives loss EMAs time to become meaningful." />
						<FormField fieldPath="training.modality_freeze_ema_decay" label="EMA Decay" type="number" value={$projectConfig?.training?.modality_freeze_ema_decay ?? 0.99} oninput={(e) => updateTraining('modality_freeze_ema_decay', Number(e.target.value))} step="0.005" min={0} max={1} tooltip="Smoothing for modality-freeze video/audio loss tracking. Default 0.99." />
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
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Advanced noise/timestep sampling controls. Safe default: Auto logit mode, Preserve Distribution Shape off, Timestep Buckets off. Change these only when matching a known recipe or debugging timestep coverage.
				</p>
			</div>

			<div class="p-5 pt-2 space-y-2">
				<div class="grid grid-cols-3 gap-2">
					<FormSelect fieldPath="training.shifted_logit_mode" label="Logit Mode" value={$projectConfig?.training?.shifted_logit_mode || ''} options={[{value:'',label:'Auto'},{value:'legacy',label:'Legacy'},{value:'stretched',label:'Stretched'}]} onchange={(e) => updateTraining('shifted_logit_mode', e.target.value || null)} tooltip="Auto uses trainer defaults. Legacy matches older behavior. Stretched matches the newer upstream shifted-logit sampler. Leave Auto unless reproducing a recipe." />
					<FormField fieldPath="training.shifted_logit_eps" label="Eps" type="number" value={$projectConfig?.training?.shifted_logit_eps ?? 0.001} oninput={(e) => updateTraining('shifted_logit_eps', Number(e.target.value))} step="0.001" tooltip="Numerical epsilon used by stretched mode. Default 0.001. Do not change unless you hit numerical edge cases." />
					<FormField fieldPath="training.shifted_logit_uniform_prob" label="Uniform Prob" type="number" value={$projectConfig?.training?.shifted_logit_uniform_prob ?? 0.1} oninput={(e) => updateTraining('shifted_logit_uniform_prob', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Probability of sampling from a uniform fallback instead of the shifted-logit distribution. Default 0.1 improves coverage." />
				</div>
				<div class="grid grid-cols-2 gap-2">
					<div class="flex items-end pb-0.5">
						<FormToggle fieldPath="training.preserve_distribution_shape" label="Preserve Distribution Shape" checked={$projectConfig?.training?.preserve_distribution_shape ?? false} onchange={(e) => updateTraining('preserve_distribution_shape', e.target.checked)} tooltip="Uses rejection sampling when min/max timesteps are set so the original distribution shape is less distorted. Default off." />
					</div>
					<FormField fieldPath="training.num_timestep_buckets" label="Timestep Buckets" type="number" value={$projectConfig?.training?.num_timestep_buckets ?? ''} oninput={(e) => updateTraining('num_timestep_buckets', e.target.value ? Number(e.target.value) : null)} placeholder="Off" min={2} tooltip="Stratifies timestep sampling into N buckets for more even coverage. Blank is off and is the safe default." />
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
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Preservation losses reduce catastrophic forgetting but add extra objectives or extra model calls. Safe default: leave these off unless you have a specific drift problem. TARP, DCR, and Audio Quality Metrics are audio/video tools; they only make sense for AV runs.
				</p>
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
						Restricts video-frame cross-attention to nearby audio tokens so audio/video alignment is more local in time. Requires AV mode. Safe default: off. Try it when the model hears the right audio but motion timing drifts.
					</p>
					<FormField fieldPath="training.tarp_window_multiplier" label="Window Multiplier" type="number" value={$projectConfig?.training?.tarp_window_multiplier ?? 3} oninput={(e) => updateTraining('tarp_window_multiplier', Number(e.target.value))} step="1" min={1} tooltip="Cross-attention window size = multiplier times audio tokens per frame. Default 3. Lower is stricter timing; higher allows looser alignment." />
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
						Dynamically detaches gradients for streams that should not learn on a given sample, such as missing-audio samples or clean reference conditioning. Requires AV mode. Safe default: off; useful for mixed AV batches where absent audio otherwise contaminates learning.
					</p>
					<FormToggle fieldPath="training.dcr_reference_detach" label="Reference Detach" checked={$projectConfig?.training?.dcr_reference_detach ?? true} onchange={(e) => updateTraining('dcr_reference_detach', e.target.checked)} tooltip="Default on when DCR is enabled. Detaches gradients when sigma=0, which usually means clean reference conditioning rather than a noisy training target." />
					<div class="mt-2">
						<FormField fieldPath="training.dcr_args" value={$projectConfig?.training?.dcr_args || ''} oninput={(e) => updateTraining('dcr_args', e.target.value)} placeholder="reference_detach=false" tooltip="Additional values passed after --dcr_args." />
					</div>
				</div>

				<!-- Audio Quality Metrics -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Audio Quality Metrics</span>
						<FormToggle fieldPath="training.audio_metrics" checked={$projectConfig?.training?.audio_metrics ?? false} onchange={(e) => updateTraining('audio_metrics', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Logs audio diagnostics during training. Basic latent metrics are cheap; Mel Metrics decode audio periodically and cost more. CLAP Similarity and AV Onset Alignment are sampling-time diagnostics, not training losses. Safe default: off unless you are actively tracking AV quality.
					</p>
					<div class="grid grid-cols-3 gap-x-4 gap-y-1 mb-2">
						<FormToggle fieldPath="training.audio_metrics_mel_metrics" label="Mel Metrics" checked={$projectConfig?.training?.audio_metrics_mel_metrics ?? false} onchange={(e) => updateTraining('audio_metrics_mel_metrics', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Computes spectral convergence, MCD, and log-spectral distance every Mel Interval steps. Requires VAE/audio decode, so leave off unless you need detailed audio diagnostics." />
						<FormToggle fieldPath="training.audio_metrics_clap_similarity" label="CLAP Similarity" checked={$projectConfig?.training?.audio_metrics_clap_similarity ?? false} onchange={(e) => updateTraining('audio_metrics_clap_similarity', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Sampling-time audio-text embedding similarity. Useful for prompt/audio relevance checks, but it may require loading a CLAP model." />
						<FormToggle fieldPath="training.audio_metrics_av_onset_alignment" label="AV Onset Alignment" checked={$projectConfig?.training?.audio_metrics_av_onset_alignment ?? false} onchange={(e) => updateTraining('audio_metrics_av_onset_alignment', e.target.checked)} disabled={!$projectConfig?.training?.audio_metrics} tooltip="Sampling-time correlation between audio onsets and video motion. Useful for rhythm/sync evaluation, not a training loss." />
					</div>
					<FormField fieldPath="training.audio_metrics_mel_compute_every" label="Mel Interval" type="number" value={$projectConfig?.training?.audio_metrics_mel_compute_every ?? 100} oninput={(e) => updateTraining('audio_metrics_mel_compute_every', Number(e.target.value))} step="10" min={1} disabled={!$projectConfig?.training?.audio_metrics_mel_metrics} tooltip="Compute mel-space metrics every N steps. Default 100. Increase to reduce overhead." />
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
