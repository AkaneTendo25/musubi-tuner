<script>
	import { tick } from 'svelte';
	import { processConsoleUi, setProcessConsoleCollapsed } from '$lib/stores/processes.js';

	let { lines = [], maxLines = 1000, initiallyCollapsed = false, processType = null, emptyMessage = 'No output yet' } = $props();

	let container = $state(null);
	let autoScroll = $state(true);
	let localCollapsed = $state(false);
	let collapsed = $derived.by(() => {
		if (processType) {
			const storedCollapsed = $processConsoleUi?.[processType]?.collapsed;
			return storedCollapsed == null ? initiallyCollapsed : storedCollapsed;
		}
		return localCollapsed;
	});

	$effect(() => {
		if (!processType) {
			localCollapsed = initiallyCollapsed;
		}
	});

	function stripAnsi(str) {
		return str.replace(/\x1b\[[0-9;]*m/g, '').replace(/\r/g, '');
	}

	function lineColor(text) {
		const t = text.toLowerCase();
		if (/^(\$|>|::)/.test(text) || /^(python|pip|accelerate)\s/.test(t)) return 'var(--accent)';
		if (/error|traceback|exception|failed|fatal/i.test(text)) return 'var(--danger)';
		if (/warn(ing)?:/i.test(text)) return 'var(--warning)';
		if (/saved|checkpoint|completed|success|done/i.test(text)) return 'var(--success)';
		if (/step\s+\d+|epoch\s+\d+|loss[=:\s]/i.test(text)) return 'var(--info)';
		return 'var(--console-text)';
	}

	$effect(() => {
		if (lines.length && autoScroll && container) {
			tick().then(() => {
				if (container) container.scrollTop = container.scrollHeight;
			});
		}
	});

	function handleScroll() {
		if (!container || collapsed) return;
		const { scrollTop, scrollHeight, clientHeight } = container;
		autoScroll = scrollHeight - scrollTop - clientHeight < 50;
	}

	let displayLines = $derived(
		collapsed ? (lines.length ? [lines[lines.length - 1]] : []) : lines.slice(-maxLines)
	);

	function toggleCollapsed() {
		const next = !collapsed;
		if (processType) {
			setProcessConsoleCollapsed(processType, next);
		} else {
			localCollapsed = next;
		}
		autoScroll = true;
		if (container) container.scrollTop = container.scrollHeight;
	}
</script>

<div class="relative">
	<div class="flex items-center justify-between gap-3 mb-2">
		<div class="text-[10px] font-medium uppercase tracking-[0.22em]" style="color: var(--text-muted); font-family: var(--font-label);">
			Console
		</div>
		<button
			onclick={toggleCollapsed}
			class="text-[10px] font-medium px-2 py-0.5"
			style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
		>{collapsed ? 'Expand log' : 'Collapse log'}</button>
	</div>

	<div
		bind:this={container}
		onscroll={handleScroll}
		class="font-mono text-[12px]"
		class:h-64={!collapsed}
		class:h-[34px]={collapsed}
		class:overflow-auto={!collapsed}
		class:overflow-hidden={collapsed}
		class:leading-5={!collapsed}
		class:leading-4={collapsed}
		class:p-3={!collapsed}
		class:px-3={collapsed}
		class:py-1.5={collapsed}
		style="background: var(--console-bg); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: inset 0 2px 6px rgba(0,0,0,.3);"
	>
		{#each displayLines as line}
			{@const clean = stripAnsi(line)}
			<div
				class:truncate={collapsed}
				class:whitespace-nowrap={collapsed}
				class:whitespace-pre-wrap={!collapsed}
				class:break-all={!collapsed}
				style="color: {lineColor(clean)};"
			>{clean}</div>
		{/each}
		{#if displayLines.length === 0}
			<div class="italic" style="color: var(--console-text-muted);">{emptyMessage}</div>
		{/if}
	</div>

	<button
		onclick={() => {
			autoScroll = true;
			if (container) container.scrollTop = container.scrollHeight;
		}}
		class="absolute top-10 right-2 text-[10px] font-medium px-2 py-0.5 transition-opacity"
		style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--accent); border-radius: var(--radius-sm); opacity: {!collapsed && !autoScroll && lines.length > 0 ? '1' : '0'}; pointer-events: {!collapsed && !autoScroll && lines.length > 0 ? 'auto' : 'none'};"
	>Scroll to bottom</button>
</div>
