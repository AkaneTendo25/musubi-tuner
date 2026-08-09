<script>
	import ComfyConversionPanel from '$lib/components/ComfyConversionPanel.svelte';
	import LoRAExtractionPanel from '$lib/components/LoRAExtractionPanel.svelte';
	import LoRAConversionPanel from '$lib/components/LoRAConversionPanel.svelte';
	import Int8ConvRotQuantizePanel from '$lib/components/Int8ConvRotQuantizePanel.svelte';
	import Int4ConvRotQuantizePanel from '$lib/components/Int4ConvRotQuantizePanel.svelte';
	import Int8WeightsExportPanel from '$lib/components/Int8WeightsExportPanel.svelte';
	import ProjectConfigPanel from '$lib/components/ProjectConfigPanel.svelte';
	import { projectLoaded } from '$lib/stores/project.js';
</script>

{#if !$projectLoaded}
	<div class="p-4 text-sm" style="color: var(--text-secondary);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-4">
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4 items-start">
			<ProjectConfigPanel />
			<LoRAExtractionPanel />
			<ComfyConversionPanel />
			<Int8ConvRotQuantizePanel />
			<Int4ConvRotQuantizePanel />
			<Int8WeightsExportPanel />
			<LoRAConversionPanel />
		</div>
	</div>
{/if}
