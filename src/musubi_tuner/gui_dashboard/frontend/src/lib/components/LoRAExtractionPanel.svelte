<script>
	import { onDestroy } from 'svelte';
	import PathInput from './PathInput.svelte';
	import { saveProjectNow } from '$lib/stores/project.js';

	let baseModelPath = $state('');
	let finetunedModelPath = $state('');
	let outputPath = $state('');
	let targetPreset = $state('full');
	let extractMode = $state('lora');
	let rankMode = $state('fro');
	let maxRank = $state(128);
	let froTarget = $state(0.98);
	let connectorLora = $state(false);
	let unsupportedTensors = $state('report');
	let jobId = $state('');
	let jobState = $state('');
	let status = $state('');
	let statusTone = $state('muted');
	let pollTimer = null;

	let active = $derived(Boolean(jobId) && ['queued', 'running'].includes(jobState));

	function setStatus(job) {
		jobState = job?.state || '';
		if (!job) {
			status = '';
			statusTone = 'muted';
			return;
		}
		if (job.state === 'completed') {
			status = job.message || (job.output_path ? `Saved to ${job.output_path}` : 'Extraction complete');
			statusTone = 'success';
		} else if (job.state === 'failed') {
			status = job.error || job.message || 'Extraction failed';
			statusTone = 'danger';
		} else if (job.state === 'running') {
			status = job.message || 'Extracting LoRA';
			statusTone = 'accent';
		} else {
			status = job.message || 'Queued';
			statusTone = 'muted';
		}
		if (job.output_path) outputPath = job.output_path;
		if (job.base_model_path) baseModelPath = job.base_model_path;
	}

	async function poll(job) {
		clearTimeout(pollTimer);
		try {
			const res = await fetch(`/api/tools/extract-lora/${job}`, { cache: 'no-store' });
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Extraction status failed');
			setStatus(data);
			if (['queued', 'running'].includes(data.state)) {
				pollTimer = setTimeout(() => poll(job), 1000);
			} else {
				jobId = '';
			}
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Extraction status failed' });
			jobId = '';
		}
	}

	async function extractLoRA() {
		if (active) return;
		if (!finetunedModelPath.trim()) {
			setStatus({ state: 'failed', error: 'Select a fine-tuned checkpoint first' });
			return;
		}
		try {
			await saveProjectNow();
			setStatus({ state: 'queued', message: 'Queued' });
			const res = await fetch('/api/tools/extract-lora', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					base_model_path: baseModelPath.trim(),
					finetuned_model_path: finetunedModelPath.trim(),
					output_path: outputPath.trim(),
					target_preset: targetPreset,
					connector_lora: connectorLora,
					extract_mode: extractMode,
					rank_mode: rankMode,
					dim: Math.max(1, Number(maxRank) || 1),
					max_rank: Math.max(1, Number(maxRank) || 1),
					fro_target: Number(froTarget) || 0.98,
					unsupported_tensors: unsupportedTensors,
					device: 'cpu'
				})
			});
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Failed to start extraction');
			jobId = data.job_id || '';
			setStatus(data);
			if (jobId) await poll(jobId);
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Failed to start extraction' });
			jobId = '';
		}
	}

	onDestroy(() => {
		clearTimeout(pollTimer);
	});
</script>

<div class="p-3.5 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
	<div class="flex items-center justify-between gap-3">
		<div>
			<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Fine-Tune Extraction</div>
			<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">Full checkpoint to LoRA</div>
		</div>
	</div>

	<PathInput
		label="Fine-tuned checkpoint"
		value={finetunedModelPath}
		oninput={(e) => finetunedModelPath = e.target.value}
		showFiles
		tooltip="Full fine-tuned LTX-2 transformer checkpoint."
		actionLabel="Extract"
		actionBusyLabel="..."
		actionDisabled={active}
		actionTooltip="Extract a native Musubi LoRA from the selected full checkpoint"
		onaction={extractLoRA}
	/>
	<PathInput
		label="Base checkpoint"
		value={baseModelPath}
		oninput={(e) => baseModelPath = e.target.value}
		showFiles
		placeholder="Auto: loaded project LTX-2 checkpoint"
		disabled={active}
		tooltip="Original base checkpoint used before full fine-tuning. Leave blank to use the loaded project checkpoint."
	/>
	<PathInput
		label="Output"
		value={outputPath}
		oninput={(e) => outputPath = e.target.value}
		showFiles
		placeholder="Auto: selected-name.extracted_lora.safetensors"
		disabled={active}
		tooltip="Native Musubi LoRA output path. A JSON report is written beside it."
	/>

	<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Target</span>
			<select bind:value={targetPreset} disabled={active} class="w-full h-8 px-2 text-[12px]" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
				<option value="full">Full</option>
				<option value="t2v">T2V</option>
				<option value="v2v">V2V</option>
				<option value="video_sa">Video self-attn</option>
				<option value="video_sa_ff">Video self-attn + FF</option>
				<option value="video_sa_ca_ff">Video SA + CA + FF</option>
				<option value="audio">Audio</option>
				<option value="audio_v2a">Audio + V2A</option>
				<option value="audio_ref_ic">Audio reference IC</option>
				<option value="av_ic">AV IC</option>
			</select>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Mode</span>
			<select bind:value={extractMode} disabled={active} class="w-full h-8 px-2 text-[12px]" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
				<option value="lora">LoRA</option>
				<option value="dora">DoRA</option>
			</select>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Rank mode</span>
			<select bind:value={rankMode} disabled={active} class="w-full h-8 px-2 text-[12px]" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
				<option value="fro">Frobenius energy</option>
				<option value="fixed">Fixed</option>
				<option value="quantile">Quantile</option>
				<option value="knee">Knee</option>
				<option value="relative_drop">Relative drop</option>
			</select>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Unsupported</span>
			<select bind:value={unsupportedTensors} disabled={active} class="w-full h-8 px-2 text-[12px]" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
				<option value="report">Report</option>
				<option value="error">Error</option>
				<option value="skip">Skip</option>
				<option value="sidecar">Sidecar</option>
			</select>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Rank / max rank</span>
			<input bind:value={maxRank} disabled={active} type="number" min="1" step="1" class="w-full h-8 px-2 text-[12px]" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);" />
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Fro target</span>
			<input bind:value={froTarget} disabled={active || rankMode !== 'fro'} type="number" min="0.5" max="0.999" step="0.001" class="w-full h-8 px-2 text-[12px]" style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);" />
		</label>
	</div>

	<label class="inline-flex items-center gap-2 text-[12px]" style="color: var(--text-secondary);">
		<input type="checkbox" bind:checked={connectorLora} disabled={active} />
		<span>Include connector linears</span>
	</label>

	{#if status}
		<div class="text-[11px] px-3 py-2 truncate" title={status} style="color: {statusTone === 'success' ? 'var(--success)' : statusTone === 'accent' ? 'var(--accent)' : statusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {statusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : statusTone === 'accent' ? 'var(--accent-muted)' : statusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);">
			{status}
		</div>
	{/if}
</div>
