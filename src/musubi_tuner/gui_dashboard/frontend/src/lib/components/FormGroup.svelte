<script>
	import { slide } from 'svelte/transition';
	let { title, collapsed = $bindable(false), children } = $props();
</script>

<div class="form-group-card" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm); position: relative; overflow: hidden;">
	<div class="form-group-accent-line" style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
	<button
		type="button"
		onclick={() => collapsed = !collapsed}
		class="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold tracking-wider transition-colors uppercase"
		style="color: var(--text-muted); font-family: var(--font-label); letter-spacing: 0.8px;"
	>
		<span>{title}</span>
		<svg class="w-3.5 h-3.5 transition-transform {collapsed ? '' : 'rotate-180'}" style="color: var(--text-muted)" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M19 9l-7 7-7-7"/></svg>
	</button>
	{#if !collapsed}
		<div transition:slide={{ duration: 200 }} class="px-4 pb-4 space-y-3" style="border-top: 1px solid var(--border-subtle);">
			{@render children()}
		</div>
	{/if}
</div>
