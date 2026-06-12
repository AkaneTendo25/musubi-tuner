<script>
	import ComfyConversionPanel from '$lib/components/ComfyConversionPanel.svelte';
	import LoRAConversionPanel from '$lib/components/LoRAConversionPanel.svelte';
	import { projectLoaded } from '$lib/stores/project.js';
</script>

{#if !$projectLoaded}
	<div class="p-4 text-sm" style="color: var(--text-secondary);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-4 max-w-4xl">
		<div>
			<h1 class="text-[18px] font-semibold" style="color: var(--text-primary);">Tools</h1>
		</div>

		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4 items-start">
			<ComfyConversionPanel />
			<LoRAConversionPanel />
		</div>
	</div>
{/if}
