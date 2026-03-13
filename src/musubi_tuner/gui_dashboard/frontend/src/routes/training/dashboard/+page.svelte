<script>
	import StatusBar from '$lib/components/StatusBar.svelte';
	import LossChart from '$lib/components/LossChart.svelte';
	import LRChart from '$lib/components/LRChart.svelte';
	import StepTimeChart from '$lib/components/StepTimeChart.svelte';
	import SampleGallery from '$lib/components/SampleGallery.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import { projectConfig } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, fetchLogs } from '$lib/stores/processes.js';
	import { dataLoading } from '$lib/stores/metrics.js';
	import { onMount } from 'svelte';

	let t = $derived($projectConfig?.training || {});

	let trainingStatus = $derived($processStatuses.training || { state: 'idle', exit_code: null });
	let trainingLogs = $derived($processLogs.training || []);
	let trainingActive = $derived(trainingStatus.state === 'running' || trainingStatus.state === 'stopping' || trainingStatus.state === 'finished');

	let systemInfo = $state(null);

	onMount(async () => {
		fetchLogs('training');
		try {
			const res = await fetch('/api/system/info');
			if (res.ok) systemInfo = await res.json();
		} catch {}
		const interval = setInterval(async () => {
			try {
				const res = await fetch('/api/system/info');
				if (res.ok) systemInfo = await res.json();
			} catch {}
		}, 2000);
		return () => clearInterval(interval);
	});

	let gpu = $derived(systemInfo?.gpus?.[0]);
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<h2 class="text-base font-semibold" style="color: var(--text-primary);">Training Dashboard</h2>
		<a href="/training" class="px-3 py-1.5 text-[12px] font-medium" style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);">Config</a>
	</div>

	<!-- Status + Hardware row -->
	{#if trainingActive}
		<StatusBar />
	{/if}

	<div class="grid grid-cols-2 xl:grid-cols-4 gap-3" style="min-height: 80px;">
		{#if systemInfo}
			{#if gpu}
				<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
					<div class="flex items-center justify-between">
						<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">VRAM</div>
						{#if gpu.temperature != null}
							<span class="text-[10px] tabular-nums" style="color: {gpu.temperature > 80 ? 'var(--danger)' : 'var(--text-muted)'};">{gpu.temperature}°C</span>
						{/if}
					</div>
					<div class="text-[14px] font-semibold mt-1 tabular-nums" style="color: var(--accent);">{(gpu.vram_used_mb / 1024).toFixed(1)} / {(gpu.vram_total_mb / 1024).toFixed(0)} GB</div>
					<div class="h-1 mt-1.5 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
						<div class="h-full" style="width: {(gpu.vram_used_mb / gpu.vram_total_mb * 100).toFixed(0)}%; background: var(--accent); border-radius: var(--radius-full); transition: width 0.6s ease;"></div>
					</div>
				</div>
				{#if gpu.utilization != null}
					<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
						<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">GPU Load</div>
						<div class="text-[14px] font-semibold mt-1 tabular-nums" style="color: var(--text-primary);">{gpu.utilization}%</div>
						<div class="h-1 mt-1.5 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
							<div class="h-full" style="width: {gpu.utilization}%; background: {gpu.utilization > 90 ? 'var(--success)' : 'var(--info)'}; border-radius: var(--radius-full); transition: width 0.6s ease;"></div>
						</div>
					</div>
				{/if}
			{/if}
			{#if systemInfo.ram}
				<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
					<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">RAM</div>
					<div class="text-[14px] font-semibold mt-1 tabular-nums" style="color: var(--text-primary);">{systemInfo.ram.used_gb} / {systemInfo.ram.total_gb} GB</div>
					<div class="h-1 mt-1.5 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
						<div class="h-full" style="width: {systemInfo.ram.percent}%; background: {systemInfo.ram.percent > 90 ? 'var(--danger)' : 'var(--info)'}; border-radius: var(--radius-full); transition: width 0.6s ease;"></div>
					</div>
				</div>
			{/if}
			{#if systemInfo.disk}
				<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
					<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Disk Free</div>
					<div class="text-[14px] font-semibold mt-1 tabular-nums" style="color: var(--text-primary);">{systemInfo.disk.free_gb} GB</div>
					<div class="h-1 mt-1.5 overflow-hidden" style="background: var(--border); border-radius: var(--radius-full);">
						<div class="h-full" style="width: {(systemInfo.disk.used_gb / systemInfo.disk.total_gb * 100).toFixed(0)}%; background: var(--info); border-radius: var(--radius-full); transition: width 0.6s ease;"></div>
					</div>
				</div>
			{/if}
		{/if}
	</div>

	<!-- Process Controls + Console — always visible -->
	<div class="space-y-3">
		<ProcessControls processType="training" status={trainingStatus} onStart={() => startProcess('training')} onStop={() => stopProcess('training')} />
		<ProcessConsole lines={trainingLogs} />
	</div>

	{#if trainingActive}
		{#if $dataLoading}
			<div class="text-center text-[12px] py-8" style="color: var(--text-muted);">Loading metrics...</div>
		{/if}

		<LossChart />

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<LRChart />
			<StepTimeChart />
		</div>

		<SampleGallery />
	{/if}
</div>
