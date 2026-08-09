<script>
	import { onDestroy } from 'svelte';
	import PathInput from './PathInput.svelte';
	import { saveProjectNow } from '$lib/stores/project.js';

	let checkpointPath = $state('');
	let outputPath = $state('');
	let groupsize = $state('auto');
	let calcDevice = $state('cpu');
	let mseClip = $state(true);
	let qualityReport = $state(true);
	let convrotPolicy = $state('');
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
			status = job.message || (job.output_path ? `Saved to ${job.output_path}` : 'Quantization complete');
			statusTone = 'success';
		} else if (job.state === 'failed') {
			status = job.error || job.message || 'Quantization failed';
			statusTone = 'danger';
		} else if (job.state === 'running') {
			status = job.message || 'Pre-quantizing to INT8 ConvRot';
			statusTone = 'accent';
		} else {
			status = job.message || 'Queued';
			statusTone = 'muted';
		}
		if (job.output_path) outputPath = job.output_path;
	}

	async function poll(job) {
		clearTimeout(pollTimer);
		try {
			const res = await fetch(`/api/tools/quantize-int8cr/${job}`, { cache: 'no-store' });
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Quantization status failed');
			setStatus(data);
			if (['queued', 'running'].includes(data.state)) {
				pollTimer = setTimeout(() => poll(job), 1000);
			} else {
				jobId = '';
			}
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Quantization status failed' });
			jobId = '';
		}
	}

	async function quantizeCheckpoint() {
		if (active) return;
		if (!checkpointPath.trim()) {
			setStatus({ state: 'failed', error: 'Select a checkpoint first' });
			return;
		}
		try {
			await saveProjectNow();
			setStatus({ state: 'queued', message: 'Queued' });
			const res = await fetch('/api/tools/quantize-int8cr', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					checkpoint_path: checkpointPath.trim(),
					output_path: outputPath.trim(),
					groupsize: groupsize.trim() || 'auto',
					mse_clip: mseClip,
					calc_device: calcDevice,
					quality_report: qualityReport,
					convrot_policy: convrotPolicy.trim()
				})
			});
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Failed to start quantization');
			jobId = data.job_id || '';
			setStatus(data);
			if (jobId) await poll(jobId);
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Failed to start quantization' });
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
			<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">INT8 ConvRot</div>
			<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">Pre-quantize checkpoint</div>
		</div>
	</div>

	<PathInput
		label="Checkpoint"
		value={checkpointPath}
		oninput={(e) => checkpointPath = e.target.value}
		showFiles
		tooltip="Base bf16/fp16 (or scaled-fp8) checkpoint. Transformer-block Linear weights are quantized to INT8 ConvRot; everything else is copied through. Load the output with --int8_convrot_base to skip on-the-fly quantization."
		actionLabel="Quantize"
		actionBusyLabel="..."
		actionDisabled={active}
		actionTooltip="Pre-quantize selected checkpoint to INT8 ConvRot"
		onaction={quantizeCheckpoint}
	/>

	<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Group size</span>
			<input
				bind:value={groupsize}
				disabled={active}
				placeholder="auto"
				class="w-full h-8 px-2 text-[12px]"
				style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
			/>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Compute device</span>
			<select
				bind:value={calcDevice}
				disabled={active}
				class="w-full h-8 px-2 text-[12px]"
				style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
			>
				<option value="cpu">CPU</option>
				<option value="cuda">CUDA (faster)</option>
			</select>
		</label>
	</div>

	<div class="flex flex-wrap items-center gap-4">
		<label class="flex items-center gap-2 text-[12px]" style="color: var(--text-secondary);">
			<input type="checkbox" bind:checked={mseClip} disabled={active} />
			MSE-optimal scale clipping
		</label>
		<label class="flex items-center gap-2 text-[12px]" style="color: var(--text-secondary);">
			<input type="checkbox" bind:checked={qualityReport} disabled={active} />
			Write quality report
		</label>
	</div>

	<PathInput
		label="ConvRot policy"
		value={convrotPolicy}
		oninput={(e) => convrotPolicy = e.target.value}
		showFiles
		placeholder="Optional policy JSON"
		disabled={active}
		tooltip="Optional per-layer policy. quantize=false rules keep matching weights in floating point."
	/>

	<PathInput
		label="Output"
		value={outputPath}
		oninput={(e) => outputPath = e.target.value}
		showFiles
		placeholder="Auto: selected-name.int8cr.safetensors"
		disabled={active}
		tooltip="Optional output path. Leave blank to save beside the selected checkpoint."
	/>

	{#if status}
		<div class="text-[11px] px-3 py-2 truncate" title={status} style="color: {statusTone === 'success' ? 'var(--success)' : statusTone === 'accent' ? 'var(--accent)' : statusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {statusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : statusTone === 'accent' ? 'var(--accent-muted)' : statusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);">
			{status}
		</div>
	{/if}
</div>
