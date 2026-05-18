<script>
	import { onMount, tick } from 'svelte';

	let visible = $state(false);
	let positioned = $state(false);
	let text = $state('');
	let left = $state(0);
	let top = $state(0);
	let placement = $state('top');
	let tooltipEl = $state(null);
	let activeTarget = null;

	const GAP = 10;
	const MARGIN = 8;

	function tooltipTargetFrom(eventTarget) {
		if (!(eventTarget instanceof Element)) return null;
		return eventTarget.closest('[data-tooltip]');
	}

	async function showFor(target) {
		const value = target?.getAttribute('data-tooltip');
		if (!value) return;

		activeTarget = target;
		text = value;
		positioned = false;
		visible = true;
		await tick();
		updatePosition();
	}

	function hide() {
		activeTarget = null;
		visible = false;
		positioned = false;
	}

	function clamp(value, min, max) {
		if (max < min) return min;
		return Math.min(Math.max(value, min), max);
	}

	function updatePosition() {
		if (!visible || !activeTarget || !tooltipEl) return;
		if (!document.body.contains(activeTarget)) {
			hide();
			return;
		}

		const targetRect = activeTarget.getBoundingClientRect();
		const tooltipRect = tooltipEl.getBoundingClientRect();
		const viewportWidth = window.innerWidth;
		const viewportHeight = window.innerHeight;

		let nextLeft = targetRect.left + targetRect.width / 2 - tooltipRect.width / 2;
		nextLeft = clamp(nextLeft, MARGIN, viewportWidth - tooltipRect.width - MARGIN);

		let nextTop = targetRect.top - tooltipRect.height - GAP;
		placement = 'top';

		if (nextTop < MARGIN) {
			nextTop = targetRect.bottom + GAP;
			placement = 'bottom';
		}
		nextTop = clamp(nextTop, MARGIN, viewportHeight - tooltipRect.height - MARGIN);

		left = nextLeft;
		top = nextTop;
		positioned = true;
	}

	function handlePointerOver(event) {
		const target = tooltipTargetFrom(event.target);
		if (target && target !== activeTarget) {
			showFor(target);
		}
	}

	function handlePointerOut(event) {
		if (!activeTarget) return;
		const related = event.relatedTarget;
		if (related instanceof Node && activeTarget.contains(related)) return;
		const target = event.target;
		if (target instanceof Node && activeTarget.contains(target)) hide();
	}

	function handleFocusIn(event) {
		const target = tooltipTargetFrom(event.target);
		if (target) showFor(target);
	}

	function handleFocusOut(event) {
		if (!activeTarget) return;
		const related = event.relatedTarget;
		if (related instanceof Node && activeTarget.contains(related)) return;
		hide();
	}

	onMount(() => {
		document.addEventListener('pointerover', handlePointerOver, true);
		document.addEventListener('pointerout', handlePointerOut, true);
		document.addEventListener('focusin', handleFocusIn, true);
		document.addEventListener('focusout', handleFocusOut, true);
		window.addEventListener('scroll', updatePosition, true);
		window.addEventListener('resize', updatePosition);

		return () => {
			document.removeEventListener('pointerover', handlePointerOver, true);
			document.removeEventListener('pointerout', handlePointerOut, true);
			document.removeEventListener('focusin', handleFocusIn, true);
			document.removeEventListener('focusout', handleFocusOut, true);
			window.removeEventListener('scroll', updatePosition, true);
			window.removeEventListener('resize', updatePosition);
		};
	});
</script>

{#if visible}
	<div
		bind:this={tooltipEl}
		class="tooltip-portal {placement === 'bottom' ? 'tooltip-portal-bottom' : ''} {positioned ? 'tooltip-portal-positioned' : ''}"
		style:left={`${left}px`}
		style:top={`${top}px`}
		role="tooltip"
	>
		{text}
	</div>
{/if}
