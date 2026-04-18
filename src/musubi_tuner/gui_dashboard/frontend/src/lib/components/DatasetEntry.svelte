<script>
	import FormField from './FormField.svelte';
	import FormSelect from './FormSelect.svelte';
	import PathInput from './PathInput.svelte';

	let { entry = {}, index = 0, onRemove, onchange, advanced = false } = $props();

	function emit(nextEntry) {
		if (onchange) {
			onchange(nextEntry);
		}
	}

	function updateField(field, value) {
		emit({ ...entry, [field]: value });
	}

	function parseNumberInput(value, optional = false) {
		if (value === '' || value === null || value === undefined) {
			return optional ? null : '';
		}

		const parsed = Number(value);
		return Number.isNaN(parsed) ? (optional ? null : '') : parsed;
	}

	function updateNumberField(field, value, optional = false) {
		updateField(field, parseNumberInput(value, optional));
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
	const isAudio = $derived(entry.type === 'audio');
	const mediaLabel = $derived(isAudio ? 'Audio' : isVideo ? 'Video' : 'Image');
	const usesReferenceDirectory = $derived(Boolean(entry.reference_cache_directory));
	const sourceDirectoryLabel = $derived(usesReferenceDirectory ? 'Reference Directory' : 'Control Directory');
	const sourceDirectoryTooltip = $derived(
		usesReferenceDirectory
			? 'Directory with reference images/videos. When Reference Cache Dir is set, this is exported as reference_directory for IC-LoRA datasets.'
			: 'Directory with control images/videos (e.g. depth maps, edges).'
	);
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

	<div class="p-4 space-y-3">
		<div class="grid grid-cols-2 gap-3">
			<FormSelect
				label="Type"
				value={entry.type}
				onchange={(e) => updateField('type', e.target.value)}
				options={typeOptions}
				tooltip="Type of media in this dataset"
			/>
			<FormField
				label="Caption Ext"
				value={entry.caption_extension}
				oninput={(e) => updateField('caption_extension', e.target.value)}
				placeholder=".txt"
				tooltip="File extension for caption files"
			/>
		</div>

		<PathInput
			label={`${mediaLabel} Directory`}
			value={entry.directory}
			oninput={(e) => updateField('directory', e.target.value)}
			tooltip="Directory containing media files"
		/>
		<PathInput
			label="JSONL File"
			value={entry.jsonl_file}
			oninput={(e) => updateField('jsonl_file', e.target.value)}
			showFiles
			placeholder="Optional (alternative to directory)"
			tooltip="JSONL file listing individual media paths instead of a directory"
		/>
		{#if advanced}
			<PathInput
				label="Cache Directory"
				value={entry.cache_directory}
				oninput={(e) => updateField('cache_directory', e.target.value)}
				tooltip="Directory to store cached latents/embeddings"
			/>
			<PathInput
				label="Reference Cache Dir"
				value={entry.reference_cache_directory}
				oninput={(e) => updateField('reference_cache_directory', e.target.value)}
				placeholder="Optional"
				tooltip="Reference cache directory for I2V / control workflows"
			/>
		{/if}
		{#if advanced && !isAudio}
			<PathInput
				label={sourceDirectoryLabel}
				value={entry.control_directory}
				oninput={(e) => updateField('control_directory', e.target.value)}
				placeholder="Optional"
				tooltip={sourceDirectoryTooltip}
			/>
		{/if}

		{#if !isAudio}
			<div class="grid grid-cols-3 gap-3">
				<FormField
					label="Width"
					type="number"
					value={entry.resolution_w}
					oninput={(e) => updateNumberField('resolution_w', e.target.value)}
					min={64}
					step={64}
					tooltip="Target width in pixels (multiples of 64)"
				/>
				<FormField
					label="Height"
					type="number"
					value={entry.resolution_h}
					oninput={(e) => updateNumberField('resolution_h', e.target.value)}
					min={64}
					step={64}
					tooltip="Target height in pixels (multiples of 64)"
				/>
				<FormField
					label="Batch Size"
					type="number"
					value={entry.batch_size}
					oninput={(e) => updateNumberField('batch_size', e.target.value)}
					min={1}
					tooltip="Training batch size for this dataset"
				/>
			</div>
		{:else}
			<FormField
				label="Batch Size"
				type="number"
				value={entry.batch_size}
				oninput={(e) => updateNumberField('batch_size', e.target.value)}
				min={1}
				tooltip="Training batch size for this dataset"
			/>
		{/if}

		<FormField
			label="Num Repeats"
			type="number"
			value={entry.num_repeats}
			oninput={(e) => updateNumberField('num_repeats', e.target.value)}
			min={1}
			tooltip="Number of times to repeat this dataset per epoch"
		/>

		{#if isVideo}
			<div class="pt-3 space-y-3" style="border-top: 1px solid var(--border-subtle);">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Video Options</span>
				<div class="grid grid-cols-2 gap-3">
					<FormField
						label="Target Frames"
						type="number"
						value={entry.target_frames}
						oninput={(e) => updateNumberField('target_frames', e.target.value)}
						min={1}
						tooltip="Number of frames to extract per video"
					/>
					<FormSelect
						label="Frame Extraction"
						value={entry.frame_extraction}
						onchange={(e) => updateField('frame_extraction', e.target.value)}
						options={extractionOptions}
						tooltip="Method used to select frames from video"
					/>
				</div>
				{#if advanced}
					<div class="grid grid-cols-3 gap-3">
						<FormField
							label="Frame Sample"
							type="number"
							value={entry.frame_sample ?? ''}
							oninput={(e) => updateNumberField('frame_sample', e.target.value, true)}
							placeholder="Optional"
							tooltip="Sample interval for frame extraction (varies by method)"
						/>
						<FormField
							label="Max Frames"
							type="number"
							value={entry.max_frames ?? ''}
							oninput={(e) => updateNumberField('max_frames', e.target.value, true)}
							placeholder="Optional"
							tooltip="Maximum frames limit per video clip"
						/>
						<FormField
							label="Frame Stride"
							type="number"
							value={entry.frame_stride ?? ''}
							oninput={(e) => updateNumberField('frame_stride', e.target.value, true)}
							placeholder="Optional"
							tooltip="Stride between extracted frames"
						/>
					</div>
					<div class="grid grid-cols-2 gap-3">
						<FormField
							label="Source FPS"
							type="number"
							value={entry.source_fps ?? ''}
							oninput={(e) => updateNumberField('source_fps', e.target.value, true)}
							placeholder="Optional"
							step="0.1"
							tooltip="Source video frame rate (for FPS-aware extraction)"
						/>
						<FormField
							label="Target FPS"
							type="number"
							value={entry.target_fps ?? ''}
							oninput={(e) => updateNumberField('target_fps', e.target.value, true)}
							placeholder="Optional"
							step="0.1"
							tooltip="Target frame rate for resampling"
						/>
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>
