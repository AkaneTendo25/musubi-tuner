<script>
	import FormField from './FormField.svelte';
	import FormSelect from './FormSelect.svelte';
	import PathInput from './PathInput.svelte';

	let { entry = $bindable({}), index = 0, onRemove, onchange } = $props();

	function setField(field, value) {
		entry[field] = value;
		entry = entry;  // reassign to trigger $bindable propagation
		if (onchange) onchange();
	}

	const typeOptions = [
		{ value: 'video', label: 'Video' },
		{ value: 'image', label: 'Image' },
		{ value: 'audio', label: 'Audio' }
	];

	const extractionOptions = [
		{ value: 'head', label: 'Head (first N)' },
		{ value: 'chunk', label: 'Chunk (consecutive)' },
		{ value: 'slide', label: 'Slide (sliding window)' },
		{ value: 'uniform', label: 'Uniform (evenly spaced)' },
		{ value: 'full', label: 'Full (all frames)' }
	];

	const isVideo = $derived(entry.type === 'video');
	const isImage = $derived(entry.type === 'image');
	const isAudio = $derived(entry.type === 'audio');
</script>

<div class="overflow-hidden" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
	<div class="flex items-center justify-between px-4 py-2.5" style="border-bottom: 1px solid var(--border-subtle);">
		<span class="text-[13px] font-semibold" style="color: var(--text-primary);">Dataset #{index + 1}</span>
		<button
			onclick={onRemove}
			class="px-2.5 py-1 text-[11px] font-medium"
			style="color: var(--danger); border-radius: var(--radius-full); background: var(--danger-muted);"
			onmouseenter={(e) => e.currentTarget.style.opacity = '0.8'}
			onmouseleave={(e) => e.currentTarget.style.opacity = '1'}
		>Remove</button>
	</div>

	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="p-4 space-y-3" oninput={() => { entry = entry; if (onchange) onchange(); }} onchange={() => { entry = entry; if (onchange) onchange(); }}>
		<div class="grid grid-cols-2 gap-3">
			<FormSelect label="Type" bind:value={entry.type} options={typeOptions} tooltip="Type of media in this dataset" />
			<FormField label="Caption Ext" bind:value={entry.caption_extension} placeholder=".txt" tooltip="File extension for caption files" />
		</div>

		<PathInput label="{isAudio ? 'Audio' : isVideo ? 'Video' : 'Image'} Directory" bind:value={entry.directory} onselect={(p) => setField('directory', p)} tooltip="Directory containing media files" />
		<PathInput label="JSONL File" bind:value={entry.jsonl_file} onselect={(p) => setField('jsonl_file', p)} showFiles placeholder="Optional (alternative to directory)" tooltip="JSONL file listing individual media paths instead of a directory" />
		<PathInput label="Cache Directory" bind:value={entry.cache_directory} onselect={(p) => setField('cache_directory', p)} tooltip="Directory to store cached latents/embeddings" />
		<PathInput label="Reference Cache Dir" bind:value={entry.reference_cache_directory} onselect={(p) => setField('reference_cache_directory', p)} placeholder="Optional" tooltip="Reference cache directory for I2V / control workflows" />
		{#if !isAudio}
			<PathInput label="Control Directory" bind:value={entry.control_directory} onselect={(p) => setField('control_directory', p)} placeholder="Optional" tooltip="Directory with control images/videos (e.g. depth maps, edges)" />
		{/if}

		{#if !isAudio}
			<div class="grid grid-cols-3 gap-3">
				<FormField label="Width" type="number" bind:value={entry.resolution_w} min={64} step={64} tooltip="Target width in pixels (multiples of 64)" />
				<FormField label="Height" type="number" bind:value={entry.resolution_h} min={64} step={64} tooltip="Target height in pixels (multiples of 64)" />
				<FormField label="Batch Size" type="number" bind:value={entry.batch_size} min={1} tooltip="Training batch size for this dataset" />
			</div>
		{:else}
			<FormField label="Batch Size" type="number" bind:value={entry.batch_size} min={1} tooltip="Training batch size for this dataset" />
		{/if}

		<FormField label="Num Repeats" type="number" bind:value={entry.num_repeats} min={1} tooltip="Number of times to repeat this dataset per epoch" />

		{#if isVideo}
			<div class="pt-3 space-y-3" style="border-top: 1px solid var(--border-subtle);">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Video Options</span>
				<div class="grid grid-cols-2 gap-3">
					<FormField label="Target Frames" type="number" bind:value={entry.target_frames} min={1} tooltip="Number of frames to extract per video" />
					<FormSelect label="Frame Extraction" bind:value={entry.frame_extraction} options={extractionOptions} tooltip="Method used to select frames from video" />
				</div>
				<div class="grid grid-cols-3 gap-3">
					<FormField label="Frame Sample" type="number" bind:value={entry.frame_sample} placeholder="Optional" tooltip="Sample interval for frame extraction (varies by method)" />
					<FormField label="Max Frames" type="number" bind:value={entry.max_frames} placeholder="Optional" tooltip="Maximum frames limit per video clip" />
					<FormField label="Frame Stride" type="number" bind:value={entry.frame_stride} placeholder="Optional" tooltip="Stride between extracted frames" />
				</div>
				<div class="grid grid-cols-2 gap-3">
					<FormField label="Source FPS" type="number" bind:value={entry.source_fps} placeholder="Optional" step="0.1" tooltip="Source video frame rate (for FPS-aware extraction)" />
					<FormField label="Target FPS" type="number" bind:value={entry.target_fps} placeholder="Optional" step="0.1" tooltip="Target frame rate for resampling" />
				</div>
			</div>
		{/if}
	</div>
</div>
