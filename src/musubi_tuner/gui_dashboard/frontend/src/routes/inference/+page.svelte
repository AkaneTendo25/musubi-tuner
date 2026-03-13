<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import CheckpointInput from '$lib/components/CheckpointInput.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, updateSection } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, fetchLogs } from '$lib/stores/processes.js';
	import { onMount } from 'svelte';

	onMount(() => { fetchLogs('inference'); });

	function update(key, value) { updateSection('inference', key, value); }

	let s = $derived($projectConfig?.inference || {});
	let inferenceStatus = $derived($processStatuses.inference || { state: 'idle', exit_code: null });
	let inferenceLogs = $derived($processLogs.inference || []);
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-4">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Inference</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">Generate videos using your trained LoRA.</p>
		</div>

		<!-- Two-column layout -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
			<!-- Left column -->
			<div class="space-y-3">
				<FormGroup title="Model">
					<div class="space-y-2 pt-2">
						<CheckpointInput label="LTX-2 Checkpoint" value={s.ltx2_checkpoint || ''} onchange={(v) => update('ltx2_checkpoint', v)} showFiles tooltip="Path to LTX-2 checkpoint" />
						<CheckpointInput label="Gemma Root" value={s.gemma_root || ''} onchange={(v) => update('gemma_root', v)} tooltip="Gemma text encoder directory" />
						<div class="grid grid-cols-3 gap-2">
							<FormSelect label="Mode" value={s.ltx2_mode || 'video'} options={['video', 'av', 'audio']} onchange={(e) => update('ltx2_mode', e.target.value)} tooltip="Video/AV/Audio" />
							<FormSelect label="Precision" value={s.mixed_precision || 'bf16'} options={['no', 'fp16', 'bf16']} onchange={(e) => update('mixed_precision', e.target.value)} tooltip="Mixed precision mode" />
							<FormSelect label="Attn Mode" value={s.attn_mode || 'torch'} options={['torch', 'xformers', 'flash', 'sage']} onchange={(e) => update('attn_mode', e.target.value)} tooltip="Attention implementation" />
						</div>
						<div class="flex flex-wrap gap-x-4 gap-y-1">
							<FormToggle label="FP8 Base" checked={s.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision (VRAM savings)" />
							<FormToggle label="FP8 Scaled" checked={s.fp8_scaled ?? false} onchange={(e) => update('fp8_scaled', e.target.checked)} tooltip="Scaled FP8 for stability" />
							<FormToggle label="Gemma 8b" checked={s.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
							<FormToggle label="Gemma 4b" checked={s.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="LoRA">
					<div class="space-y-2 pt-2">
						<PathInput label="LoRA Weight" value={s.lora_weight || ''} oninput={(e) => update('lora_weight', e.target.value)} showFiles tooltip="Path to LoRA safetensors file" />
						<FormField label="Multiplier" type="number" value={s.lora_multiplier ?? 1.0} oninput={(e) => update('lora_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="LoRA weight multiplier" />
					</div>
				</FormGroup>

				<FormGroup title="Prompt">
					<div class="space-y-2 pt-2">
						<!-- svelte-ignore a11y_label_has_associated_control -->
						<label class="block">
							<span class="block text-[11px] font-medium mb-1" style="color: var(--text-muted);">Prompt</span>
							<textarea
								class="w-full text-[12px] px-3 py-2 resize-y"
								rows="3"
								value={s.prompt || ''}
								oninput={(e) => update('prompt', e.target.value)}
								placeholder="Describe what you want to generate..."
								style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
								onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
								onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
							></textarea>
						</label>
						<!-- svelte-ignore a11y_label_has_associated_control -->
						<label class="block">
							<span class="block text-[11px] font-medium mb-1" style="color: var(--text-muted);">Negative Prompt</span>
							<textarea
								class="w-full text-[12px] px-3 py-2 resize-y"
								rows="2"
								value={s.negative_prompt || ''}
								oninput={(e) => update('negative_prompt', e.target.value)}
								placeholder="Optional negative prompt..."
								style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
								onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
								onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
							></textarea>
						</label>
						<PathInput label="Prompts File" value={s.from_file || ''} oninput={(e) => update('from_file', e.target.value)} showFiles tooltip="Text file with prompts (one per line, overrides prompt field)" />
					</div>
				</FormGroup>
			</div>

			<!-- Right column -->
			<div class="space-y-3">
				<FormGroup title="Generation">
					<div class="space-y-2 pt-2">
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Width" type="number" value={s.width ?? 768} oninput={(e) => update('width', Number(e.target.value))} min={64} step={64} tooltip="Output width" />
							<FormField label="Height" type="number" value={s.height ?? 512} oninput={(e) => update('height', Number(e.target.value))} min={64} step={64} tooltip="Output height" />
							<FormField label="Frames" type="number" value={s.frame_count ?? 45} oninput={(e) => update('frame_count', Number(e.target.value))} min={1} tooltip="Number of frames" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Frame Rate" type="number" value={s.frame_rate ?? 25} oninput={(e) => update('frame_rate', Number(e.target.value))} min={1} step={1} tooltip="Frames per second" />
							<FormField label="Steps" type="number" value={s.sample_steps ?? 20} oninput={(e) => update('sample_steps', Number(e.target.value))} min={1} tooltip="Sampling steps" />
							<FormField label="Seed" type="number" value={s.seed ?? ''} oninput={(e) => update('seed', e.target.value ? Number(e.target.value) : null)} placeholder="Random" tooltip="Random seed" />
						</div>
						<div class="grid grid-cols-3 gap-2">
							<FormField label="Guidance" type="number" value={s.guidance_scale ?? 1.0} oninput={(e) => update('guidance_scale', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance scale" />
							<FormField label="CFG Scale" type="number" value={s.cfg_scale ?? ''} oninput={(e) => update('cfg_scale', e.target.value ? Number(e.target.value) : null)} placeholder="Optional" step="0.1" tooltip="Classifier-free guidance (optional)" />
							<FormField label="Flow Shift" type="number" value={s.discrete_flow_shift ?? 5.0} oninput={(e) => update('discrete_flow_shift', Number(e.target.value))} step="0.1" tooltip="Discrete flow shift" />
						</div>
					</div>
				</FormGroup>

				<FormGroup title="Memory">
					<div class="space-y-2 pt-2">
						<FormToggle label="Offloading" checked={s.offloading ?? false} onchange={(e) => update('offloading', e.target.checked)} tooltip="Enable model offloading to reduce VRAM" />
						<FormField label="Blocks to Swap" type="number" value={s.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0-40" min={0} max={40} tooltip="CPU offload blocks" />
					</div>
				</FormGroup>

				<FormGroup title="Output">
					<div class="space-y-2 pt-2">
						<PathInput label="Output Dir" value={s.output_dir || 'output'} oninput={(e) => update('output_dir', e.target.value)} tooltip="Output directory" />
						<FormField label="Name" value={s.output_name || 'ltx2_sample'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix" />
					</div>
				</FormGroup>
			</div>
		</div>

		<!-- Controls -->
		<div class="py-4">
			<ProcessControls processType="inference" status={inferenceStatus} onStart={() => startProcess('inference')} onStop={() => stopProcess('inference')} />
		</div>

		<ProcessConsole lines={inferenceLogs} />
		<CommandPanel processType="inference" defaultFilename="inference.bat" />
	</div>
{/if}
