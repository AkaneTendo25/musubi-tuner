<script>
	import { onMount } from 'svelte';
	import { projectConfig } from '$lib/stores/project.js';

	let stats = $state(null);
	let loading = $state(true);
	let error = $state(null);

	async function fetchStats() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/stats');
			if (res.ok) {
				stats = await res.json();
			} else {
				error = 'Failed to load stats';
			}
		} catch (e) {
			error = e.message;
		}
		loading = false;
	}

	// Debounced refetch when config changes (prevent too frequent API calls)
	let lastConfigSnapshot = '';
	let refetchTimer = null;
	$effect(() => {
		const cfg = $projectConfig;
		if (cfg) {
			const snapshot = JSON.stringify(cfg);
			if (snapshot !== lastConfigSnapshot) {
				lastConfigSnapshot = snapshot;
				// Debounce: wait 2 seconds after last change
				if (refetchTimer) clearTimeout(refetchTimer);
				refetchTimer = setTimeout(fetchStats, 2000);
			}
		}
	});

	onMount(() => {
		fetchStats();
	});

	function formatNumber(num) {
		if (num === null || num === undefined) return 'N/A';
		return num.toLocaleString(undefined, { maximumFractionDigits: 1 });
	}

	function formatTime(hours) {
		if (!hours) return 'N/A';
		if (hours < 1) return `${Math.round(hours * 60)} min`;
		if (hours < 24) return `${hours.toFixed(1)} hrs`;
		return `${(hours / 24).toFixed(1)} days`;
	}

	const statRowClass = 'flex items-baseline gap-2 flex-wrap';
</script>

<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm); position: relative; overflow: hidden;">
	<!-- Accent gradient -->
	<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>

	<div class="px-3 py-2 flex items-center justify-between" style="border-bottom: 1px solid var(--border-subtle);">
		<span class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Project Statistics</span>
		{#if loading}
			<svg class="w-3 h-3 animate-spin" style="color: var(--text-muted);" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
		{/if}
	</div>

	{#if error}
		<div class="p-3 text-center text-[11px]" style="color: var(--danger);">
			{error}
		</div>
	{:else if loading && !stats}
		<div class="p-3 space-y-2 animate-pulse" style="min-height: 80px;">
			<div class="h-2 rounded" style="background: var(--border); width: 60%;"></div>
			<div class="h-2 rounded" style="background: var(--border); width: 80%;"></div>
			<div class="h-2 rounded" style="background: var(--border); width: 50%;"></div>
		</div>
	{:else if stats && !stats.dataset}
		<!-- No dataset configured - show VRAM only or compact message -->
		<div class="p-3">
			{#if stats.vram}
				<div class="p-2.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
					<div class="text-[10px] font-semibold mb-1.5 uppercase tracking-wider" style="color: var(--accent);">VRAM Estimates</div>
					<div class="space-y-1.5">
						<div class={statRowClass}>
							<span class="text-[11px]" style="color: var(--text-muted);">Peak Training:</span>
							<span class="text-[13px] font-bold" style="color: var(--success);">{formatNumber(stats.vram.peak_training_gb)} GB</span>
						</div>
						<div class={statRowClass}>
							<span class="text-[11px]" style="color: var(--text-muted);">Peak Sampling:</span>
							<span class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.vram.peak_sampling_gb)} GB</span>
						</div>
					</div>
				</div>
				<div class="mt-2 text-center text-[10px]" style="color: var(--text-muted);">
					<svg class="w-3.5 h-3.5 inline opacity-50 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"/>
					</svg>
					Configure dataset for training statistics
				</div>
			{:else}
				<div class="text-center text-[11px]" style="color: var(--text-muted); min-height: 60px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
					<svg class="w-4 h-4 mb-1 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
						<path d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"/>
					</svg>
					Configure a dataset to see statistics
				</div>
			{/if}
		</div>
	{:else if stats}
		<div class="p-3 space-y-3 animate-fadeIn" style="animation: fadeIn 0.3s ease-in;">
			<!-- Dataset Stats -->
			{#if stats.dataset}
				<div class="p-2.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
					<div class="text-[10px] font-semibold mb-1.5 uppercase tracking-wider" style="color: var(--accent);">Dataset</div>
					<div class="space-y-1.5">
						<div class={statRowClass}>
							<div class="text-[11px]" style="color: var(--text-muted);">Total Items:</div>
							<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.dataset.total_items)}</div>
						</div>

						<div class={statRowClass}>
							<div class="text-[11px]" style="color: var(--text-muted);">Video:</div>
							<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.dataset.video_items)}</div>
						</div>

						{#if stats.dataset.audio_items > 0}
							<div class={statRowClass}>
								<div class="text-[11px]" style="color: var(--text-muted);">Audio:</div>
								<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.dataset.audio_items)}</div>
							</div>
						{/if}

						{#if stats.dataset.max_resolution}
							<div class={statRowClass}>
								<div class="text-[11px]" style="color: var(--text-muted);">Max Resolution:</div>
								<div class="text-[11px] font-medium font-mono" style="color: var(--text-primary);">{stats.dataset.max_resolution[0]}×{stats.dataset.max_resolution[1]}</div>
							</div>
						{/if}

						{#if stats.dataset.max_frames}
							<div class={statRowClass}>
								<div class="text-[11px]" style="color: var(--text-muted);">Max Frames:</div>
								<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.dataset.max_frames)}</div>
							</div>
						{/if}
					</div>
				</div>
			{/if}

			<!-- Training Stats -->
			{#if stats.training}
				<div class="p-2.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
					<div class="text-[10px] font-semibold mb-1.5 uppercase tracking-wider" style="color: var(--accent);">Training</div>
					<div class="space-y-1.5">
						<div class={statRowClass}>
							<div class="text-[11px]" style="color: var(--text-muted);">Effective Batch Size:</div>
							<div class="text-[11px] font-medium" style="color: var(--text-primary);">{stats.training.effective_batch_size}</div>
						</div>

						{#if stats.training.steps_per_epoch}
							<div class={statRowClass}>
								<div class="text-[11px]" style="color: var(--text-muted);">Steps per Epoch:</div>
								<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.training.steps_per_epoch)}</div>
							</div>
						{/if}

						{#if stats.training.total_epochs}
							<div class={statRowClass}>
								<div class="text-[11px]" style="color: var(--text-muted);">Total Epochs:</div>
								<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.training.total_epochs)}</div>
							</div>
						{/if}

						{#if stats.training.estimated_time_hours}
							<div class={statRowClass}>
								<div class="text-[11px]" style="color: var(--text-muted);">Est. Duration:</div>
								<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatTime(stats.training.estimated_time_hours)}</div>
							</div>
						{/if}

						<div class={statRowClass}>
							<div class="text-[11px]" style="color: var(--text-muted);">Checkpoints:</div>
							<div class="text-[11px] font-medium" style="color: var(--text-primary);">{stats.training.total_checkpoints}</div>
						</div>

						<div class={statRowClass}>
							<div class="text-[11px]" style="color: var(--text-muted);">Storage (LoRA):</div>
							<div class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.training.total_storage_gb)} GB</div>
						</div>
					</div>
				</div>
			{/if}

			<!-- VRAM Stats -->
			{#if stats.vram}
				<div class="p-2.5" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
					<div class="text-[10px] font-semibold mb-1.5 uppercase tracking-wider" style="color: var(--accent);">VRAM Estimates</div>
					<div class="space-y-1.5">
						<div class={statRowClass}>
							<span class="text-[11px]" style="color: var(--text-muted);">Peak Training:</span>
							<span class="text-[13px] font-bold" style="color: var(--success);">{formatNumber(stats.vram.peak_training_gb)} GB</span>
						</div>
						<div class={statRowClass}>
							<span class="text-[11px]" style="color: var(--text-muted);">Peak Sampling:</span>
							<span class="text-[11px] font-medium" style="color: var(--text-primary);">{formatNumber(stats.vram.peak_sampling_gb)} GB</span>
						</div>

						<!-- Breakdown -->
						<details class="mt-1">
							<summary class="text-[10px] cursor-pointer" style="color: var(--text-muted);">Show breakdown</summary>
							<div class="mt-1.5 pl-2 space-y-0.5 border-l-2" style="border-color: var(--border);">
								<div class={statRowClass + ' text-[10px]'}>
									<span style="color: var(--text-muted);">Model:</span>
									<span style="color: var(--text-secondary);">{stats.vram.breakdown.model} GB</span>
								</div>
								<div class={statRowClass + ' text-[10px]'}>
									<span style="color: var(--text-muted);">Optimizer:</span>
									<span style="color: var(--text-secondary);">{stats.vram.breakdown.optimizer} GB</span>
								</div>
								<div class={statRowClass + ' text-[10px]'}>
									<span style="color: var(--text-muted);">Activations:</span>
									<span style="color: var(--text-secondary);">{stats.vram.breakdown.activations} GB</span>
								</div>
								<div class={statRowClass + ' text-[10px]'}>
									<span style="color: var(--text-muted);">Overhead:</span>
									<span style="color: var(--text-secondary);">{stats.vram.breakdown.overhead} GB</span>
								</div>
							</div>
						</details>
					</div>
				</div>
			{/if}

			{#if !stats.dataset && !stats.training && !stats.vram}
				<div class="p-3 text-center text-[11px]" style="color: var(--text-muted);">
					Configure your project to see statistics
				</div>
			{/if}
		</div>
		{/if}
</div>

<style>
	@keyframes fadeIn {
		from { opacity: 0; transform: translateY(-4px); }
		to { opacity: 1; transform: translateY(0); }
	}
</style>
