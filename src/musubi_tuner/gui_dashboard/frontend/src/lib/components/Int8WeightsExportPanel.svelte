<script>
	import { onDestroy } from 'svelte';
	import PathInput from './PathInput.svelte';
	import { saveProjectNow } from '$lib/stores/project.js';

	let checkpointPath = $state('');
	let outputPath = $state('');
	let targets = $state('video');
	let groupSize = $state('0');
	let convrot = $state('auto');
	let dtype = $state('bfloat16');
	let calcDevice = $state('cpu');
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
			status = job.message || (job.output_path ? `Saved to ${job.output_path}` : 'Pre-quantization complete');
			statusTone = 'success';
		} else if (job.state === 'failed') {
			status = job.error || job.message || 'Pre-quantization failed';
			statusTone = 'danger';
		} else if (job.state === 'running') {
			status = job.message || 'Pre-quantizing to INT8 weight-only grid';
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
			const res = await fetch(`/api/tools/quantize-int8w/${job}`, { cache: 'no-store' });
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Pre-quantization status failed');
			setStatus(data);
			if (['queued', 'running'].includes(data.state)) {
				pollTimer = setTimeout(() => poll(job), 1000);
			} else {
				jobId = '';
			}
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Pre-quantization status failed' });
			jobId = '';
		}
	}

	async function exportCheckpoint() {
		if (active) return;
		if (!checkpointPath.trim()) {
			setStatus({ state: 'failed', error: 'Select a checkpoint first' });
			return;
		}
		try {
			await saveProjectNow();
			setStatus({ state: 'queued', message: 'Queued' });
			const res = await fetch('/api/tools/quantize-int8w', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					checkpoint_path: checkpointPath.trim(),
					output_path: outputPath.trim(),
					targets: targets.trim() || 'video',
					group_size: groupSize.trim() || '0',
					convrot: convrot.trim() || 'auto',
					dtype: dtype,
					calc_device: calcDevice
				})
			});
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Failed to start pre-quantization');
			jobId = data.job_id || '';
			setStatus(data);
			if (jobId) await poll(jobId);
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Failed to start pre-quantization' });
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
			<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">INT8 weight-only (FFT)</div>
			<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">Pre-quantize for --int8_weights_prequant</div>
		</div>
	</div>

	<PathInput
		label="Checkpoint"
		value={checkpointPath}
		oninput={(e) => checkpointPath = e.target.value}
		showFiles
		tooltip="Base bf16/fp16 (or scaled-fp8) checkpoint. Target-block Linear weights are quantized to the INT8 weight-only (Int8QTWeight) grid so full fine-tuning with --int8_weights can skip the startup quantization. Load the output with --int8_weights_prequant. The grid fields below MUST match your --int8_weights_* training flags."
		actionLabel="Pre-quantize"
		actionBusyLabel="..."
		actionDisabled={active}
		actionTooltip="Pre-quantize selected checkpoint to the INT8 weight-only grid"
		onaction={exportCheckpoint}
	/>

	<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Targets</span>
			<input
				bind:value={targets}
				disabled={active}
				placeholder="video"
				class="w-full h-8 px-2 text-[12px]"
				style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
			/>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Group size</span>
			<input
				bind:value={groupSize}
				disabled={active}
				placeholder="0"
				class="w-full h-8 px-2 text-[12px]"
				style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
			/>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">ConvRot</span>
			<input
				bind:value={convrot}
				disabled={active}
				placeholder="auto"
				class="w-full h-8 px-2 text-[12px]"
				style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
			/>
		</label>
		<label class="space-y-1.5">
			<span class="block text-[11px] font-medium" style="color: var(--text-secondary); font-family: var(--font-label);">Weight dtype</span>
			<select
				bind:value={dtype}
				disabled={active}
				class="w-full h-8 px-2 text-[12px]"
				style="background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
			>
				<option value="bfloat16">bfloat16</option>
				<option value="float16">float16</option>
				<option value="float32">float32</option>
			</select>
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

	<PathInput
		label="Output"
		value={outputPath}
		oninput={(e) => outputPath = e.target.value}
		showFiles
		placeholder="Auto: selected-name.int8w.safetensors"
		disabled={active}
		tooltip="Optional output path. Leave blank to save beside the selected checkpoint."
	/>

	{#if status}
		<div class="text-[11px] px-3 py-2 truncate" title={status} style="color: {statusTone === 'success' ? 'var(--success)' : statusTone === 'accent' ? 'var(--accent)' : statusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {statusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : statusTone === 'accent' ? 'var(--accent-muted)' : statusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);">
			{status}
		</div>
	{/if}
</div>
