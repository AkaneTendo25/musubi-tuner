<script>
	import '../app.css';
	import { onMount, onDestroy } from 'svelte';
	import { connectSSE, disconnectSSE } from '$lib/stores/sse.js';
	import { startStatusPolling, stopStatusPolling } from '$lib/stores/status.js';
	import { startMetricsPolling, stopMetricsPolling } from '$lib/stores/metrics.js';
	import { loadProject } from '$lib/stores/project.js';
	import { connectProcessSSE, disconnectProcessSSE } from '$lib/stores/processes.js';
	import Sidebar from '$lib/components/Sidebar.svelte';

	let { children } = $props();

	onMount(async () => {
		connectSSE();
		startStatusPolling();
		startMetricsPolling();
		await loadProject();
		connectProcessSSE();
	});

	onDestroy(() => {
		disconnectSSE();
		stopStatusPolling();
		stopMetricsPolling();
		disconnectProcessSSE();
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
