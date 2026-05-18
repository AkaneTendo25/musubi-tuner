<script>
	import '../app.css';
	import { onMount, onDestroy } from 'svelte';
	import { connectSSE, disconnectSSE } from '$lib/stores/sse.js';
	import { startStatusPolling, stopStatusPolling } from '$lib/stores/status.js';
	import { startMetricsPolling, stopMetricsPolling } from '$lib/stores/metrics.js';
	import { loadProject, loadProjectDefaults } from '$lib/stores/project.js';
	import { connectProcessSSE, disconnectProcessSSE, connectProcessValidationAutoRefresh, disconnectProcessValidationAutoRefresh } from '$lib/stores/processes.js';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import TooltipPortal from '$lib/components/TooltipPortal.svelte';

	let { children } = $props();

	onMount(async () => {
		connectSSE();
		startStatusPolling();
		startMetricsPolling();
		await loadProjectDefaults();
		await loadProject();
		connectProcessSSE();
		connectProcessValidationAutoRefresh();
	});

	onDestroy(() => {
		disconnectSSE();
		stopStatusPolling();
		stopMetricsPolling();
		disconnectProcessSSE();
		disconnectProcessValidationAutoRefresh();
	});
</script>

<div class="h-screen flex overflow-hidden" style="background: var(--bg-base);">
	<Sidebar />
	<div class="flex-1 flex flex-col min-w-0 overflow-hidden">
		<main class="flex-1 overflow-auto p-5">
			{@render children()}
		</main>
	</div>
</div>

<TooltipPortal />
