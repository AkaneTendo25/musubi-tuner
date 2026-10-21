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
	let tooltipStateRevision = 0;
	let mobileNavigationOpen = $state(false);
	let mobileCloseButton;
	let mobileNavigation;
	const h3BlockedRoutes = new Set([
		'/samples',
		'/settings',
		'/tools',
		'/training/conditioning',
		'/training/full-finetune',
		'/training/rl'
	]);

	function commitTooltipState(nextState) {
		const revision = ++tooltipStateRevision;
		queueMicrotask(() => {
			if (revision === tooltipStateRevision) tooltipState = nextState;
		});
	}

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
		commitTooltipState({
			visible: true,
			text,
			left,
			top: above ? rect.top - 8 : rect.bottom + 8,
			placement: above ? 'top' : 'bottom'
		});
	}

	function hideTooltip() {
		tooltipTarget = null;
		commitTooltipState({ visible: false, text: '', left: 0, top: 0, placement: 'top' });
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

	function prepareMobileNavigation() {
		if (window.innerWidth >= 640 || !mobileNavigation) return;
		mobileNavigation.popover = 'auto';
		if (mobileCloseButton) mobileCloseButton.autofocus = true;
	}

	function handleNavigationClick(event) {
		if (
			mobileNavigation?.matches(':popover-open') &&
			event.target instanceof Element &&
			event.target.closest('a[href]')
		) {
			mobileNavigation.hidePopover();
		}
	}

	function handleMobileNavigationToggle(event) {
		mobileNavigationOpen = event.newState === 'open';
	}

	function syncMobileNavigationMode() {
		if (!mobileNavigation) return;
		if (window.innerWidth < 640) {
			mobileNavigation.popover = 'auto';
			return;
		}
		if (mobileNavigation.matches(':popover-open')) mobileNavigation.hidePopover();
		mobileNavigation.removeAttribute('popover');
		if (mobileCloseButton) mobileCloseButton.autofocus = false;
		mobileNavigationOpen = false;
	}

	function handleWindowResize() {
		updateTooltipPosition();
		syncMobileNavigationMode();
	}

	onMount(async () => {
		window.addEventListener('pointerover', handleTooltipOver, true);
		window.addEventListener('pointerout', handleTooltipOut, true);
		window.addEventListener('focusin', handleTooltipOver, true);
		window.addEventListener('focusout', hideTooltip, true);
		window.addEventListener('scroll', updateTooltipPosition, true);
		window.addEventListener('resize', handleWindowResize);
		mobileNavigation?.addEventListener('click', handleNavigationClick);
		syncMobileNavigationMode();
		connectSSE();
		startStatusPolling();
		startMetricsPolling();
		await loadProjectDefaults();
		await loadProject();
		if (h3BlockedRoutes.has(window.location.pathname)) {
			window.location.replace('/');
			return;
		}
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
			window.removeEventListener('resize', handleWindowResize);
			mobileNavigation?.removeEventListener('click', handleNavigationClick);
		}
		disconnectSSE();
		stopStatusPolling();
		stopMetricsPolling();
		disconnectProcessSSE();
		disconnectProcessValidationAutoRefresh();
	});
</script>

<div class="app-shell h-screen flex overflow-hidden" style="background: var(--bg-base);">
	<div
		id="mobile-navigation"
		class="sidebar-shell"
		bind:this={mobileNavigation}
		ontoggle={handleMobileNavigationToggle}
	>
		<button
			type="button"
			class="mobile-navigation-close"
			bind:this={mobileCloseButton}
			aria-label="Close navigation"
			popovertarget="mobile-navigation"
			popovertargetaction="hide"
		>
			<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
				<path d="M6 6l12 12M18 6L6 18" />
			</svg>
		</button>
		<Sidebar />
	</div>
	<div class="content-shell flex-1 flex flex-col min-w-0 overflow-hidden">
		<div class="mobile-navigation-bar">
			<button
				type="button"
				aria-label="Open navigation"
				aria-controls="mobile-navigation"
				aria-expanded={mobileNavigationOpen}
				popovertarget="mobile-navigation"
				popovertargetaction="show"
				onclick={prepareMobileNavigation}
			>
				<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
					<path d="M4 7h16M4 12h16M4 17h16" />
				</svg>
				<span>Menu</span>
			</button>
		</div>
		<TopStatusBar />
		<main class="app-main flex-1 min-w-0 overflow-auto p-5">
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

<style>
	.sidebar-shell {
		flex: 0 0 auto;
	}

	.mobile-navigation-bar,
	.mobile-navigation-close {
		display: none;
	}

	@media (max-width: 639px) {
		.app-shell,
		.content-shell,
		.app-main {
			width: 100%;
			max-width: 100vw;
			min-width: 0;
		}

		.sidebar-shell {
			display: none;
			position: fixed;
			top: 0;
			bottom: 0;
			left: 0;
			right: auto;
			width: 13rem;
			height: 100dvh;
			max-width: none;
			max-height: none;
			margin: 0;
			padding: 0;
			overflow: visible;
			border: 0;
		}

		.sidebar-shell:popover-open {
			display: block;
		}

		.sidebar-shell :global(.h3-sidebar) {
			height: 100dvh;
		}

		.sidebar-shell::backdrop {
			background: rgb(0 0 0 / 58%);
		}

		.mobile-navigation-close {
			position: absolute;
			top: 0.5rem;
			left: calc(100% + 0.5rem);
			display: flex;
			width: 2.25rem;
			height: 2.25rem;
			align-items: center;
			justify-content: center;
			color: var(--text-primary);
			background: var(--bg-elevated);
			border: 1px solid var(--border);
			border-radius: var(--radius-sm);
			box-shadow: var(--shadow-lg);
		}

		.mobile-navigation-close svg,
		.mobile-navigation-bar svg {
			width: 1rem;
			height: 1rem;
		}

		.mobile-navigation-bar {
			display: flex;
			flex: 0 0 auto;
			height: 2.75rem;
			align-items: center;
			padding: 0 0.75rem;
			background: var(--bg-base);
			border-bottom: 1px solid var(--border-subtle);
		}

		.mobile-navigation-bar button {
			display: inline-flex;
			align-items: center;
			gap: 0.5rem;
			padding: 0.4rem 0.65rem;
			color: var(--text-primary);
			background: var(--bg-surface);
			border: 1px solid var(--border-subtle);
			border-radius: var(--radius-sm);
			font-size: 0.75rem;
			font-weight: 600;
		}

		.app-main {
			padding: 0.75rem;
		}
	}
</style>
