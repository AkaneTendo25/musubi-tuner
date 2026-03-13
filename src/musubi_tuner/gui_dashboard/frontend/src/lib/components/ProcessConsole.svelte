<script>
	import { onMount, tick } from 'svelte';

	let { lines = [], maxLines = 1000 } = $props();

	let container = $state(null);
	let autoScroll = $state(true);

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
		if (!container) return;
		const { scrollTop, scrollHeight, clientHeight } = container;
		autoScroll = scrollHeight - scrollTop - clientHeight < 50;
	}

	let displayLines = $derived(lines.slice(-maxLines));
</script>

<div class="relative">
	<div
		bind:this={container}
		onscroll={handleScroll}
		class="font-mono text-[12px] leading-5 overflow-auto h-64 p-3"
		style="background: var(--console-bg); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: inset 0 2px 6px rgba(0,0,0,.3);"
	>
		{#each displayLines as line, i}
			{@const clean = stripAnsi(line)}
			<div class="whitespace-pre-wrap break-all" style="color: {lineColor(clean)};">{clean}</div>
		{/each}
		{#if displayLines.length === 0}
			<div class="italic" style="color: var(--console-text-muted);">No output yet</div>
		{/if}
	</div>

	<!-- Scroll to bottom — absolutely positioned so no layout jump -->
	<button
		onclick={() => { autoScroll = true; if (container) container.scrollTop = container.scrollHeight; }}
		class="absolute top-2 right-2 text-[10px] font-medium px-2 py-0.5 transition-opacity"
		style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--accent); border-radius: var(--radius-sm); opacity: {!autoScroll && lines.length > 0 ? '1' : '0'}; pointer-events: {!autoScroll && lines.length > 0 ? 'auto' : 'none'};"
	>Scroll to bottom</button>
</div>
