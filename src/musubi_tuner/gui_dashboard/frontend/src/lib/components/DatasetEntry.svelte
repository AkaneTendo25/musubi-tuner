<script>
	import FormField from './FormField.svelte';
	import FormSelect from './FormSelect.svelte';
	import FormToggle from './FormToggle.svelte';
	import PathInput from './PathInput.svelte';

	let { entry = {}, index = 0, onRemove, onchange, advanced = false, sourceError = '' } = $props();

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
	const usesReferenceDirectory = $derived(
		Boolean(entry.reference_cache_directory || entry.extra_reference_cache_directories)
	);
	const sourceDirectoryLabel = $derived(usesReferenceDirectory ? 'Reference Directory' : 'Control Directory');
	const extraSourceDirectoryLabel = $derived(
		usesReferenceDirectory ? 'Extra Reference Dirs' : 'Extra Control Dirs'
	);
	const sourceDirectoryTooltip = $derived(
		usesReferenceDirectory
			? 'Directory with reference images/videos. When Reference Cache Dir is set, this is exported as reference_directory for IC-LoRA datasets.'
			: 'Directory with control images/videos (e.g. depth maps, edges).'
	);
	const extraSourceDirectoryTooltip = $derived(
		usesReferenceDirectory
			? 'Optional extra reference image/video directories, separated by commas or semicolons.'
			: 'Optional extra control directories, separated by commas or semicolons.'
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
				tooltip="Caption file suffix for directory datasets. Use .target.txt for alternate caption files."
			/>
		</div>
		{#if advanced}
			<FormField
				label="Caption Field"
				value={entry.caption_field || ''}
				oninput={(e) => updateField('caption_field', e.target.value)}
				placeholder="caption"
				tooltip="JSONL caption key. Leave blank to use caption; set target_caption for alternate target captions."
			/>
		{/if}

		<PathInput
			label={`${mediaLabel} Directory`}
			value={entry.directory}
			oninput={(e) => updateField('directory', e.target.value)}
			tooltip="Directory containing media files"
			invalid={Boolean(sourceError)}
			error={sourceError}
		/>
		<PathInput
			label="JSONL File"
			value={entry.jsonl_file}
			oninput={(e) => updateField('jsonl_file', e.target.value)}
			showFiles
			placeholder="Optional (alternative to directory)"
			tooltip="JSONL file listing individual media paths instead of a directory"
			invalid={Boolean(sourceError)}
			error={sourceError}
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
			<PathInput
				label="Extra Ref Cache Dirs"
				value={entry.extra_reference_cache_directories}
				oninput={(e) => updateField('extra_reference_cache_directories', e.target.value)}
				placeholder="Optional"
				tooltip="Optional extra reference cache directories, separated by commas or semicolons."
			/>
			{#if !isAudio}
				<FormField
					label="Reference Frames"
					type="number"
					value={entry.reference_frames ?? ''}
					oninput={(e) => updateNumberField('reference_frames', e.target.value, true)}
					min={1}
					placeholder="Global"
					tooltip="Optional reference video frame count for this dataset. Leave blank to use the global caching Reference Frames setting."
				/>
			{/if}
			<PathInput
				label="Ref Audio Cache Dir"
				value={entry.reference_audio_cache_directory}
				oninput={(e) => updateField('reference_audio_cache_directory', e.target.value)}
				placeholder="Optional"
				tooltip="Reference audio latent cache directory for audio-ref / AV IC modes."
			/>
			<PathInput
				label="Extra Ref Audio Cache Dirs"
				value={entry.extra_reference_audio_cache_directories}
				oninput={(e) => updateField('extra_reference_audio_cache_directories', e.target.value)}
				placeholder="Optional"
				tooltip="Optional extra reference audio cache directories, separated by commas or semicolons."
			/>
			<div class="pt-3 space-y-3" style="border-top: 1px solid var(--border-subtle);">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Masked Loss</span>
				<PathInput
					label="Mask Directory"
					value={entry.loss_mask_directory || ''}
					oninput={(e) => updateField('loss_mask_directory', e.target.value)}
					placeholder="Optional"
					tooltip="Stem-matched loss masks. Image/video datasets use image/video masks; audio datasets use JSON/TXT/CSV interval files."
				/>
				<PathInput
					label="Default Mask"
					value={entry.default_loss_mask_path || ''}
					oninput={(e) => updateField('default_loss_mask_path', e.target.value)}
					showFiles
					placeholder="Optional"
					tooltip="Fallback mask used when no per-item mask is found. For audio, use an interval file."
				/>
				<div class="grid grid-cols-2 gap-3">
					<FormToggle
						label="Use Alpha"
						checked={entry.loss_mask_use_alpha ?? false}
						onchange={(e) => updateField('loss_mask_use_alpha', e.target.checked)}
						disabled={isAudio}
						tooltip="Use the target image alpha channel as the mask when no mask directory is set."
					/>
					<FormToggle
						label="Invert"
						checked={entry.loss_mask_invert ?? false}
						onchange={(e) => updateField('loss_mask_invert', e.target.checked)}
						disabled={isAudio}
						tooltip="Invert image/video mask values before caching."
					/>
				</div>
			</div>
		{/if}
		{#if advanced && !isAudio}
			<PathInput
				label={sourceDirectoryLabel}
				value={entry.control_directory}
				oninput={(e) => updateField('control_directory', e.target.value)}
				placeholder="Optional"
				tooltip={sourceDirectoryTooltip}
			/>
			<PathInput
				label={extraSourceDirectoryLabel}
				value={entry.extra_control_directories}
				oninput={(e) => updateField('extra_control_directories', e.target.value)}
				placeholder="Optional"
				tooltip={extraSourceDirectoryTooltip}
			/>
			<PathInput
				label="Ref Audio Directory"
				value={entry.reference_audio_directory}
				oninput={(e) => updateField('reference_audio_directory', e.target.value)}
				placeholder="Optional"
				tooltip="Directory with reference audio files for audio-ref / AV IC modes."
			/>
			<PathInput
				label="Extra Ref Audio Dirs"
				value={entry.extra_reference_audio_directories}
				oninput={(e) => updateField('extra_reference_audio_directories', e.target.value)}
				placeholder="Optional"
				tooltip="Optional extra reference audio directories, separated by commas or semicolons."
			/>

			<div class="pt-3 space-y-3" style="border-top: 1px solid var(--border-subtle);">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Latent Guides</span>
				<p class="text-[11px]" style="color: var(--text-muted);">
					Reference latents stem-matched to each item. <strong>Latent-idx (hard lock)</strong> replaces tokens at a frame slot — exact pixel match;
					<strong>keyframe (soft guide)</strong> appends a token the model is guided toward but not constrained to. Both compose with any
					<code>--ic_lora_strategy</code>.
				</p>
				<PathInput
					label="Latent-Idx Guide Dir"
					value={entry.latent_idx_guide_directory || ''}
					oninput={(e) => updateField('latent_idx_guide_directory', e.target.value)}
					placeholder="Optional"
					tooltip="Directory of guide images to inject at a specific latent-frame slot inside the temporal grid (e.g. last-frame anchor)."
				/>
				<PathInput
					label="Latent-Idx Guide Cache Dir"
					value={entry.latent_idx_guide_cache_directory || ''}
					oninput={(e) => updateField('latent_idx_guide_cache_directory', e.target.value)}
					placeholder="Optional"
					tooltip="Cache directory for encoded latent-idx guide latents. Required when Latent-Idx Guide Dir is set."
				/>
				<div class="grid grid-cols-2 gap-3">
					<FormField
						label="Frame Idx (latent_idx)"
						type="number"
						value={entry.latent_idx_guide_frame_idx ?? 0}
						oninput={(e) => updateNumberField('latent_idx_guide_frame_idx', e.target.value)}
						tooltip="Latent-frame slot replaced with the guide. 0 = first slot (I2V-style), or any in-grid index."
					/>
					<FormField
						label="Strength (latent_idx)"
						type="number"
						value={entry.latent_idx_guide_strength ?? 1.0}
						oninput={(e) => updateNumberField('latent_idx_guide_strength', e.target.value)}
						min={0}
						max={1}
						step={0.05}
						tooltip="Training requires 1.0 (raises if other). Inference accepts any value via the 5D denoise_mask = 1 − strength. Out-of-range values are clamped with a warning."
					/>
				</div>
				<PathInput
					label="Keyframe Guide Dir"
					value={entry.keyframe_guide_directory || ''}
					oninput={(e) => updateField('keyframe_guide_directory', e.target.value)}
					placeholder="Optional"
					tooltip="Directory of global-reference / keyframe images. Tokens are appended outside the temporal grid (frame_idx=-1 = global subject reference)."
				/>
				<PathInput
					label="Keyframe Guide Cache Dir"
					value={entry.keyframe_guide_cache_directory || ''}
					oninput={(e) => updateField('keyframe_guide_cache_directory', e.target.value)}
					placeholder="Optional"
					tooltip="Cache directory for encoded keyframe guide latents. Required when Keyframe Guide Dir is set."
				/>
				<div class="grid grid-cols-2 gap-3">
					<FormField
						label="Frame Idx (keyframe)"
						type="number"
						value={entry.keyframe_guide_frame_idx ?? -1}
						oninput={(e) => updateNumberField('keyframe_guide_frame_idx', e.target.value)}
						tooltip="Pixel-frame index (NOT latent-frame). -1 = global reference. For non-negative values multiply the latent-frame by VIDEO_SCALE_FACTORS.time first (×8 for LTX-2): e.g. last latent frame of a 9-frame video → 64."
					/>
					<FormField
						label="Strength (keyframe)"
						type="number"
						value={entry.keyframe_guide_strength ?? 1.0}
						oninput={(e) => updateNumberField('keyframe_guide_strength', e.target.value)}
						min={0}
						max={1}
						step={0.05}
						tooltip="Per-token denoise_mask = 1 − strength → effective appended timestep = (1−strength)×sigma. 1.0 = clean conditioning (default); 0.0 = full noise (no contribution). Range [0, 1]; clamped if outside."
					/>
				</div>

				<p class="text-[11px]" style="color: var(--text-muted);">
					Optional: stack extra keyframes beyond the primary above. Use <code>;</code> to separate values; all four lists must have the same length.
				</p>
				<FormField
					label="Extra Keyframe Dirs"
					value={entry.keyframe_guide_extra_directories || ''}
					oninput={(e) => updateField('keyframe_guide_extra_directories', e.target.value)}
					placeholder="/path/k2;/path/k3"
					tooltip="Semicolon-separated list of additional keyframe directories. Each one stem-matches per item like the primary."
				/>
				<FormField
					label="Extra Keyframe Cache Dirs"
					value={entry.keyframe_guide_extra_cache_directories || ''}
					oninput={(e) => updateField('keyframe_guide_extra_cache_directories', e.target.value)}
					placeholder="/cache/k2;/cache/k3"
					tooltip="Semicolon-separated list of cache directories matching the extra keyframe directories above."
				/>
				<div class="grid grid-cols-2 gap-3">
					<FormField
						label="Extra Frame Idxs"
						value={entry.keyframe_guide_extra_frame_idxs || ''}
						oninput={(e) => updateField('keyframe_guide_extra_frame_idxs', e.target.value)}
						placeholder="-1;5"
						tooltip="Semicolon-separated frame_idx values (PIXEL-frame units, not latent-frame). -1 = global reference. Multiply latent-frame by VIDEO_SCALE_FACTORS.time (×8 for LTX-2) for non-negative anchors."
					/>
					<FormField
						label="Extra Strengths"
						value={entry.keyframe_guide_extra_strengths || ''}
						oninput={(e) => updateField('keyframe_guide_extra_strengths', e.target.value)}
						placeholder="1.0;0.7"
						tooltip="Semicolon-separated strength values per extra keyframe. Range [0, 1]."
					/>
				</div>
			</div>
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
