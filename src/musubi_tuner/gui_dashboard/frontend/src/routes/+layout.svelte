<script>
	import '../app.css';
	import { onMount, onDestroy } from 'svelte';
	import { connectSSE, disconnectSSE } from '$lib/stores/sse.js';
	import { startStatusPolling, stopStatusPolling } from '$lib/stores/status.js';
	import { startMetricsPolling, stopMetricsPolling } from '$lib/stores/metrics.js';
	import { loadProject, loadProjectDefaults } from '$lib/stores/project.js';
	import { connectProcessSSE, disconnectProcessSSE, connectProcessValidationAutoRefresh, disconnectProcessValidationAutoRefresh } from '$lib/stores/processes.js';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import TopStatusBar from '$lib/components/TopStatusBar.svelte';

	let { children } = $props();
	let tooltipState = $state({ visible: false, text: '', left: 0, top: 0, placement: 'top' });
	let tooltipTarget = null;

	function updateTooltipPosition() {
		if (typeof window === 'undefined') return;
		if (tooltipTarget && !tooltipTarget.isConnected) {
			hideTooltip();
			return;
		}
		if (!tooltipTarget) return;
		const text = tooltipTarget.getAttribute('data-tooltip') || '';
		const rect = tooltipTarget.getBoundingClientRect();
		const margin = 10;
		const estimatedWidth = Math.min(260, Math.max(80, text.length * 6));
		const left = Math.min(
			Math.max(rect.left + rect.width / 2, margin + estimatedWidth / 2),
			window.innerWidth - margin - estimatedWidth / 2
		);
		const above = rect.top > 56;
		tooltipState = {
			visible: true,
			text,
			left,
			top: above ? rect.top - 8 : rect.bottom + 8,
			placement: above ? 'top' : 'bottom'
		};
	}

	function hideTooltip() {
		tooltipTarget = null;
		tooltipState = { visible: false, text: '', left: 0, top: 0, placement: 'top' };
	}

	function handleTooltipOver(event) {
		const target = event.target instanceof Element ? event.target.closest('[data-tooltip]') : null;
		if (!target) return;
		const text = target.getAttribute('data-tooltip');
		if (!text) return;
		tooltipTarget = target;
		updateTooltipPosition();
	}

	function handleTooltipOut(event) {
		if (!tooltipTarget) return;
		const related = event.relatedTarget;
		if (related instanceof Node && tooltipTarget.contains(related)) return;
		hideTooltip();
	}

	onMount(async () => {
		window.addEventListener('pointerover', handleTooltipOver, true);
		window.addEventListener('pointerout', handleTooltipOut, true);
		window.addEventListener('focusin', handleTooltipOver, true);
		window.addEventListener('focusout', hideTooltip, true);
		window.addEventListener('scroll', updateTooltipPosition, true);
		window.addEventListener('resize', updateTooltipPosition);
		connectSSE();
		startStatusPolling();
		startMetricsPolling();
		await loadProjectDefaults();
		await loadProject();
		connectProcessSSE();
		connectProcessValidationAutoRefresh();
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') {
			window.removeEventListener('pointerover', handleTooltipOver, true);
			window.removeEventListener('pointerout', handleTooltipOut, true);
			window.removeEventListener('focusin', handleTooltipOver, true);
			window.removeEventListener('focusout', hideTooltip, true);
			window.removeEventListener('scroll', updateTooltipPosition, true);
			window.removeEventListener('resize', updateTooltipPosition);
		}
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
		<TopStatusBar />
		<main class="flex-1 overflow-auto p-5">
			{@render children()}
		</main>
	</div>
</div>

{#if tooltipState.visible}
	<div
		class="global-tooltip"
		style="left: {tooltipState.left}px; top: {tooltipState.top}px; transform: translate(-50%, {tooltipState.placement === 'top' ? '-100%' : '0'});"
	>
		{tooltipState.text}
	</div>
{/if}
