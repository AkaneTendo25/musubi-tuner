<script>
	import PathInput from './PathInput.svelte';
	import { saveProjectNow } from '$lib/stores/project.js';

	let checkpointPath = $state('');
	let outputPath = $state('');
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
			status = job.message || (job.output_path ? `Saved to ${job.output_path}` : 'Conversion complete');
			statusTone = 'success';
		} else if (job.state === 'failed') {
			status = job.error || job.message || 'Conversion failed';
			statusTone = 'danger';
		} else if (job.state === 'running') {
			status = job.message || 'Converting checkpoint';
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
			const res = await fetch(`/api/tools/convert-comfy/${job}`, { cache: 'no-store' });
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Conversion status failed');
			setStatus(data);
			if (['queued', 'running'].includes(data.state)) {
				pollTimer = setTimeout(() => poll(job), 1000);
			} else {
				jobId = '';
			}
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Conversion status failed' });
			jobId = '';
		}
	}

	async function convertCheckpoint() {
		if (active) return;
		if (!checkpointPath.trim()) {
			setStatus({ state: 'failed', error: 'Select a checkpoint first' });
			return;
		}
		try {
			await saveProjectNow();
			setStatus({ state: 'queued', message: 'Queued' });
			const res = await fetch('/api/tools/convert-comfy', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					checkpoint_path: checkpointPath.trim(),
					output_path: outputPath.trim(),
					device: 'cpu'
				})
			});
			const data = await res.json().catch(() => ({}));
			if (!res.ok) throw new Error(data.detail || 'Failed to start conversion');
			jobId = data.job_id || '';
			setStatus(data);
			if (jobId) await poll(jobId);
		} catch (e) {
			setStatus({ state: 'failed', error: e?.message || 'Failed to start conversion' });
			jobId = '';
		}
	}
</script>

<div class="p-3.5 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
	<div class="flex items-center justify-between gap-3">
		<div>
			<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">ComfyUI Conversion</div>
			<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">Checkpoint export</div>
		</div>
	</div>

	<PathInput
		label="Checkpoint"
		value={checkpointPath}
		oninput={(e) => checkpointPath = e.target.value}
		showFiles
		tooltip="Musubi checkpoint to convert. Uses the loaded project's LTX-2 checkpoint and quantization settings."
		actionLabel="Convert"
		actionBusyLabel="..."
		actionDisabled={active}
		actionTooltip="Convert selected checkpoint to ComfyUI format"
		onaction={convertCheckpoint}
	/>
	<PathInput
		label="Output"
		value={outputPath}
		oninput={(e) => outputPath = e.target.value}
		showFiles
		placeholder="Auto: selected-name.comfy.safetensors"
		disabled={active}
		tooltip="Optional output path. Leave blank to save beside the selected checkpoint."
	/>

	{#if status}
		<div class="text-[11px] px-3 py-2 truncate" title={status} style="color: {statusTone === 'success' ? 'var(--success)' : statusTone === 'accent' ? 'var(--accent)' : statusTone === 'danger' ? 'var(--danger)' : 'var(--text-secondary)'}; background: {statusTone === 'success' ? 'var(--success-muted, rgba(34,197,94,0.1))' : statusTone === 'accent' ? 'var(--accent-muted)' : statusTone === 'danger' ? 'var(--danger-muted)' : 'var(--bg-elevated)'}; border-radius: var(--radius-sm);">
			{status}
		</div>
	{/if}
</div>
