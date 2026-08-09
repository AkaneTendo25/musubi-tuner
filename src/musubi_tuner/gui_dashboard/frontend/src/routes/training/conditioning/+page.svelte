<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import { projectConfig, projectLoaded, updateSection } from '$lib/stores/project.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import {
		detectConditioningObjective,
		emptyConditioningModality as emptyModality,
		emptyConditioningRecipe as emptyRecipe,
		getConditioningIssues,
		getConditioningSource,
		normalizeConditioningRecipe,
		recipeHasActive,
	} from '$lib/utils/conditioningStatus.js';

	function update(key, value) { updateSection('training', key, value); }

	let t = $derived($projectConfig?.training || {});
	let datasets = $derived($projectConfig?.dataset?.datasets || []);
	let pendingPresetId = $state(null);

	// The recipe is stored structurally in training.conditioning_recipe; default-shape it for safety.
	let recipe = $derived(normalizeConditioningRecipe(t));

	function writeRecipe(next) { update('conditioning_recipe', next); }
	function clone(r) { return JSON.parse(JSON.stringify(r)); }
	function condition(type, overrides = {}) {
		return {
			type,
			probability: null,
			invert: false,
			threshold: 0.5,
			prefix: type === 'extend' ? 1 : 0,
			suffix: 0,
			prefix_p: null,
			suffix_p: null,
			...overrides,
		};
	}

	const VIDEO_TYPES = ['first_frame', 'spatial_crop', 'inpaint', 'extend', 'reference'];
	const AUDIO_TYPES = ['extend', 'inpaint', 'reference'];
	const TYPE_LABEL = {
		first_frame: 'Start from first frame',
		spatial_crop: 'Keep a rectangle',
		inpaint: 'Use a mask',
		extend: 'Use prefix / suffix',
		reference: 'Use reference identity/style',
	};
	const SHORT_TYPE_LABEL = {
		first_frame: 'First frame',
		spatial_crop: 'Rectangle',
		inpaint: 'Mask',
		extend: 'Prefix / suffix',
		reference: 'Reference',
	};
	const TYPE_HELP = {
		first_frame: 'The model receives frame 0 clean at timestep 0, excludes it from loss, and learns the rest of the target video.',
		spatial_crop: 'The model receives a clean rectangular region from each dataset item and learns the surrounding or selected area.',
		inpaint: 'Mask > threshold is clean conditioning and excluded from loss; the remaining tokens are generated. Invert swaps keep/generate sides.',
		extend: 'The model receives clean prefix or suffix latent frames as temporal context and learns the missing continuation.',
		reference: 'The model receives separate reference tokens. This selects the IC-LoRA reference strategy automatically.',
	};
	const TYPE_RECEIVES = {
		first_frame: 'Clean target frame 0',
		spatial_crop: 'Clean dataset crop region',
		inpaint: 'Clean mask > threshold tokens',
		extend: 'Clean prefix/suffix frames',
		reference: 'Separate reference tokens',
	};
	const TYPE_LEARNS = {
		first_frame: 'Generate later video frames',
		spatial_crop: 'Generate outside/inside the region',
		inpaint: 'Generate unmasked or masked area',
		extend: 'Generate the missing continuation',
		reference: 'Match target while using reference context',
	};
	const TYPE_DATASET_NEEDS = {
		first_frame: 'Target video latents',
		spatial_crop: 'spatial_crop_region in dataset metadata/config',
		inpaint: 'loss_mask_directory or default_loss_mask_path',
		extend: 'Enough target latent frames for prefix/suffix boundary',
		reference: 'reference cache/directory for video; reference audio cache/directory for audio',
	};

	const PRESETS = [
		{
			id: 't2v',
			label: 'T2V',
			title: 'Text to video',
			blurb: 'Plain video training with no extra conditioning recipe.',
			sets: ['Mode video', 'Video generated', 'No conditions'],
			updates: { ltx2_mode: 'video', lora_target_preset: 't2v', ic_lora_strategy: 'auto', ltx2_first_frame_conditioning_p: 0 },
			recipe: emptyRecipe(),
		},
		{
			id: 'i2v',
			label: 'I2V',
			title: 'Image to video',
			blurb: 'Video training with the first video frame provided clean.',
			sets: ['Mode video', 'Video first-frame guide'],
			updates: { ltx2_mode: 'video', lora_target_preset: 't2v', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: true, conditions: [condition('first_frame', { probability: 1.0 })] }, audio: emptyModality() },
		},
		{
			id: 'video_extension',
			label: 'Video Extension',
			title: 'Continue a clip',
			blurb: 'Video extension: prefix/suffix video frames are provided clean and the missing continuation is trained.',
			sets: ['Mode video', 'Video prefix/suffix guide'],
			updates: { ltx2_mode: 'video', lora_target_preset: 't2v', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: true, conditions: [condition('extend', { prefix: 8, suffix: 0, probability: 1.0 })] }, audio: emptyModality() },
		},
		{
			id: 'v2v_ic',
			label: 'V2V IC-LoRA',
			title: 'Reference video',
			blurb: 'Train with a reference video/image stream for identity, style, or subject transfer.',
			sets: ['Mode video', 'Video reference strategy'],
			updates: { ltx2_mode: 'video', lora_target_preset: 'v2v', ic_lora_strategy: 'v2v' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: true, conditions: [condition('reference', { probability: 1.0 })] }, audio: emptyModality() },
		},
		{
			id: 'video_inpainting',
			label: 'Video Inpainting',
			title: 'Fill masked video',
			blurb: 'Keep the masked video area clean and train the model to generate the rest.',
			sets: ['Mode video', 'Video mask guide'],
			updates: { ltx2_mode: 'video', lora_target_preset: 't2v', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: true, conditions: [condition('inpaint', { probability: 1.0 })] }, audio: emptyModality() },
		},
		{
			id: 'video_outpainting',
			label: 'Video Outpainting',
			title: 'Expand around a crop',
			blurb: 'Keep a rectangular crop clean and train the surrounding video content.',
			sets: ['Mode video', 'Spatial crop guide'],
			updates: { ltx2_mode: 'video', lora_target_preset: 't2v', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: true, conditions: [condition('spatial_crop', { probability: 1.0 })] }, audio: emptyModality() },
		},
		{
			id: 'a2v',
			label: 'A2V',
			title: 'Audio to video',
			blurb: 'Freeze audio as clean conditioning and train the video stream.',
			sets: ['Mode AV', 'Audio frozen', 'Video generated'],
			updates: { ltx2_mode: 'av', lora_target_preset: 't2v', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: emptyModality(), audio: { is_generated: false, conditions: [] } },
		},
		{
			id: 'v2a',
			label: 'V2A',
			title: 'Video to audio',
			blurb: 'Freeze video as clean conditioning and train the audio or foley stream.',
			sets: ['Mode AV', 'Video frozen', 'Audio generated'],
			updates: { ltx2_mode: 'av', lora_target_preset: 'audio_v2a', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: false, conditions: [] }, audio: emptyModality() },
		},
		{
			id: 't2a',
			label: 'T2A',
			title: 'Text to audio',
			blurb: 'Generate the full audio target from text only.',
			sets: ['Mode audio', 'Audio generated'],
			updates: { ltx2_mode: 'audio', lora_target_preset: 'audio', ic_lora_strategy: 'auto' },
			recipe: emptyRecipe(),
		},
		{
			id: 'audio_extension',
			label: 'Audio Extension',
			title: 'Continue audio',
			blurb: 'Keep a clean audio prefix or suffix and train the continuation.',
			sets: ['Mode audio', 'Audio prefix guide'],
			updates: { ltx2_mode: 'audio', lora_target_preset: 'audio', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: emptyModality(), audio: { is_generated: true, conditions: [condition('extend', { prefix: 8, suffix: 0, probability: 1.0 })] } },
		},
		{
			id: 'audio_inpainting',
			label: 'Audio Inpainting',
			title: 'Fill masked audio',
			blurb: 'Keep masked audio clean and train the model to generate the missing region.',
			sets: ['Mode audio', 'Audio mask guide'],
			updates: { ltx2_mode: 'audio', lora_target_preset: 'audio', ic_lora_strategy: 'auto' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: emptyModality(), audio: { is_generated: true, conditions: [condition('inpaint', { probability: 1.0 })] } },
		},
		{
			id: 'a2a',
			label: 'A2A',
			title: 'Reference audio',
			blurb: 'Train audio generation with a separate reference audio stream.',
			sets: ['Mode audio', 'Audio reference strategy'],
			updates: { ltx2_mode: 'audio', lora_target_preset: 'audio', ic_lora_strategy: 'audio_ref_ic' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: emptyModality(), audio: { is_generated: true, conditions: [condition('reference')] } },
		},
		{
			id: 'av2av_ic',
			label: 'AV2AV IC-LoRA',
			title: 'Reference audio-video',
			blurb: 'Train video and audio together with separate reference streams for both.',
			sets: ['Mode AV', 'Video reference', 'Audio reference'],
			updates: { ltx2_mode: 'av', lora_target_preset: 'av_ic', ic_lora_strategy: 'av_ic' },
			recipe: { enabled: true, per_sample_loss: 'auto', video: { is_generated: true, conditions: [condition('reference', { probability: 1.0 })] }, audio: { is_generated: true, conditions: [condition('reference')] } },
		},
	];

	function applyPreset(preset) {
		const nextRecipe = clone(preset.recipe || emptyRecipe());
		for (const [key, value] of Object.entries(preset.updates || {})) update(key, value);
		writeRecipe(nextRecipe);
	}
	function newCondition(type) { return condition(type); }
	function conditionTypesFor(modality) { return modality === 'video' ? VIDEO_TYPES : AUDIO_TYPES; }
	function conditionIndex(modality, type) {
		return (recipe[modality].conditions || []).findIndex((c) => c.type === type);
	}
	function isConditionActive(modality, type) { return conditionIndex(modality, type) >= 0; }
	function addCondition(modality, type) {
		const r = clone(recipe);
		if ((r[modality].conditions || []).some((c) => c.type === type)) {
			return;
		}
		r[modality].conditions = [...r[modality].conditions, newCondition(type)];
		r.enabled = true;
		writeRecipe(r);
	}
	function toggleCondition(modality, type) {
		if (isConditionActive(modality, type)) removeConditionByType(modality, type);
		else addCondition(modality, type);
	}
	function removeCondition(modality, idx) {
		const r = clone(recipe);
		r[modality].conditions = r[modality].conditions.filter((_, i) => i !== idx);
		writeRecipe(r);
	}
	function removeConditionByType(modality, type) {
		const idx = conditionIndex(modality, type);
		if (idx >= 0) removeCondition(modality, idx);
	}
	function setCond(modality, idx, key, value) {
		const r = clone(recipe);
		r[modality].conditions[idx][key] = value;
		writeRecipe(r);
	}
	function setGenerated(modality, value) {
		const r = clone(recipe);
		r[modality].is_generated = value;
		// Freezing a modality is itself an activating action (directional training); without this a
		// freeze-only recipe stays inactive and silently trains plain joint.
		if (!value) r.enabled = true;
		writeRecipe(r);
	}
	function setPSL(value) { const r = clone(recipe); r.per_sample_loss = value; writeRecipe(r); }
	function setMode(mode) {
		const previousMode = t.ltx2_mode || 'video';
		if (mode === previousMode) return;
		update('ltx2_mode', mode);
	}

	function numOrNull(v) { return v === '' || v === null || v === undefined ? null : Number(v); }

	// active = a real recipe (>=1 condition or a frozen modality). Mirrors the backend gate.
	let videoModeOn = $derived(['av', 'video'].includes(t.ltx2_mode || 'video'));
	let audioModeOn = $derived(['av', 'audio'].includes(t.ltx2_mode));
	let showAudioRow = $derived(audioModeOn);
	let appliedRecipe = $derived.by(() => {
		const next = {
			enabled: recipe.enabled,
			per_sample_loss: recipe.per_sample_loss,
			video: videoModeOn ? recipe.video : emptyModality(),
			audio: audioModeOn ? recipe.audio : emptyModality(),
		};
		next.enabled = recipeHasActive(next);
		return next;
	});
	let recipeActive = $derived(recipeHasActive(appliedRecipe));
	let hasVideoRef = $derived(appliedRecipe.video.conditions.some((c) => c.type === 'reference'));
	let hasAudioRef = $derived(appliedRecipe.audio.conditions.some((c) => c.type === 'reference'));
	let isAvIc = $derived(hasVideoRef && hasAudioRef);
	let objectiveInfo = $derived(detectConditioningObjective(t));
	let selectedPresetId = $derived(pendingPresetId || (PRESETS.some((p) => p.id === objectiveInfo.id) ? objectiveInfo.id : ''));
	let selectedPreset = $derived(PRESETS.find((p) => p.id === selectedPresetId));
	let canApplySelectedPreset = $derived(!!selectedPreset);
	let conditioningSource = $derived(getConditioningSource($projectConfig));
	let conditioningIssues = $derived(getConditioningIssues($projectConfig));
	let recipeIssues = $derived(conditioningIssues.all);
	function applySelectedPreset() {
		if (!selectedPreset) return;
		applyPreset(selectedPreset);
		pendingPresetId = null;
	}

	function condLines(name, c) {
		const out = [`[[${name}.conditions]]`, `type = "${c.type}"`];
		const p = (k, v) => out.push(`${k} = ${v}`);
		if (c.type === 'extend') {
			p('prefix', Number(c.prefix || 0)); p('suffix', Number(c.suffix || 0));
			if (c.probability != null) p('probability', Number(c.probability));
			if (c.prefix_p != null) p('prefix_p', Number(c.prefix_p));
			if (c.suffix_p != null) p('suffix_p', Number(c.suffix_p));
		} else if (c.type === 'inpaint') {
			if (c.probability != null) p('probability', Number(c.probability));
			if (c.invert) p('invert', 'true');
			if (Number(c.threshold) !== 0.5) p('threshold', Number(c.threshold));
		} else if (c.type === 'spatial_crop') {
			if (c.probability != null) p('probability', Number(c.probability));
			if (c.invert) p('invert', 'true');
		} else if (c.probability != null && !(name === 'audio' && c.type === 'reference')) p('probability', Number(c.probability));
		out.push('');
		return out;
	}
	let previewToml = $derived.by(() => {
		const r = appliedRecipe;
		const lines = [];
		if (r.per_sample_loss === 'on') lines.push('per_sample_loss = true', '');
		else if (r.per_sample_loss === 'off') lines.push('per_sample_loss = false', '');
		for (const name of ['video', 'audio']) {
			const mod = r[name];
			if (mod.conditions.length === 0 && mod.is_generated) continue;
			lines.push(`[${name}]`);
			if (!mod.is_generated) lines.push('is_generated = false');
			lines.push('');
			for (const c of mod.conditions) lines.push(...condLines(name, c));
		}
		return lines.join('\n').trim() || '# (no active conditioning recipe)';
	});

	function modalityVerb(name, mod) {
		if (!mod.is_generated) return `${name} is clean input`;
		if (mod.conditions.length === 0) return `${name} is fully generated`;
		return `${name} is generated with ${mod.conditions.length} guide${mod.conditions.length === 1 ? '' : 's'}`;
	}
	let recipeSentence = $derived.by(() => {
		const parts = [];
		if (videoModeOn) parts.push(modalityVerb('Video', appliedRecipe.video));
		if (showAudioRow) parts.push(modalityVerb('Audio', appliedRecipe.audio));
		const conds = [
			...(videoModeOn ? appliedRecipe.video.conditions.map((c) => `video ${SHORT_TYPE_LABEL[c.type] || c.type}`) : []),
			...(showAudioRow ? appliedRecipe.audio.conditions.map((c) => `audio ${SHORT_TYPE_LABEL[c.type] || c.type}`) : []),
		];
		if (!recipeActive) return `${parts.join('; ')}. No active conditioning recipe is exported.`;
		return `${parts.join('; ')}. ${conds.length ? `Guides: ${conds.join(', ')}.` : 'No extra guide rows; one modality is frozen as conditioning.'} Loss is computed on generated regions.`;
	});

	function timelineSegments(modality) {
		const mod = appliedRecipe[modality];
		const segs = [];
		if (!mod.is_generated) return [{ tone: 'clean', label: 'Clean input', detail: 'no loss' }];
		const extend = mod.conditions.find((c) => c.type === 'extend');
		if (modality === 'video' && mod.conditions.some((c) => c.type === 'first_frame')) segs.push({ tone: 'clean', label: 'First frame', detail: 'clean' });
		if (extend && Number(extend.prefix || 0) > 0) segs.push({ tone: 'clean', label: `${extend.prefix} prefix`, detail: 'clean' });
		if (mod.conditions.some((c) => ['spatial_crop', 'inpaint'].includes(c.type))) segs.push({ tone: 'mask', label: 'Mask/region', detail: 'clean' });
		segs.push({ tone: 'generated', label: 'Generated target', detail: 'trained' });
		if (extend && Number(extend.suffix || 0) > 0) segs.push({ tone: 'clean', label: `${extend.suffix} suffix`, detail: 'clean' });
		if (mod.conditions.some((c) => c.type === 'reference')) segs.push({ tone: 'reference', label: 'Reference', detail: 'appended' });
		return segs;
	}

	let readinessItems = $derived.by(() => {
		const items = [];
		const needsVideoRef = videoModeOn && hasVideoRef;
		const needsAudioRef = audioModeOn && hasAudioRef;
		const needsMask = appliedRecipe.video.conditions.some((c) => c.type === 'inpaint') || (audioModeOn && appliedRecipe.audio.conditions.some((c) => c.type === 'inpaint'));
		const needsDataset = recipeActive || needsVideoRef || needsAudioRef || needsMask;
		const anyDataset = datasets.length > 0;
		if (needsDataset) {
			items.push({ label: 'Dataset configured', ok: anyDataset, detail: anyDataset ? `${datasets.length} training dataset${datasets.length === 1 ? '' : 's'}` : 'Add a dataset on the Dataset tab.' });
		}
		if (needsMask) {
			const ok = datasets.some((d) => d.loss_mask_directory || d.default_loss_mask_path);
			items.push({ label: 'Mask source', ok, detail: ok ? 'Mask path is present on at least one dataset.' : 'Set loss_mask_directory or default_loss_mask_path on the Dataset tab.' });
		}
		if (appliedRecipe.video.conditions.some((c) => c.type === 'spatial_crop')) {
			items.push({ label: 'Crop region', ok: true, detail: 'Crop region is per-dataset metadata/config; verify it is present before launch.' });
		}
		if (needsVideoRef) {
			const ok = datasets.some((d) => d.reference_cache_directory || d.reference_cache_directories || d.extra_reference_cache_directories);
			items.push({ label: 'Video reference data', ok, detail: ok ? 'Reference cache/directory is present.' : 'Set reference cache/directory on the Dataset tab.' });
		}
		if (needsAudioRef) {
			const ok = datasets.some((d) => d.reference_audio_cache_directory || d.extra_reference_audio_cache_directories || d.reference_audio_directory || d.extra_reference_audio_directories);
			items.push({ label: 'Audio reference data', ok, detail: ok ? 'Reference audio source is present.' : 'Set reference audio cache/directory on the Dataset tab.' });
		}
		if (audioModeOn && (appliedRecipe.audio.conditions.length > 0 || !appliedRecipe.audio.is_generated)) {
			items.push({ label: 'Audio-capable mode', ok: audioModeOn, detail: audioModeOn ? `Mode is ${t.ltx2_mode}.` : 'Set Mode to av or audio.' });
		}
		return items;
	});
	let recipeErrorCount = $derived(recipeIssues.filter((issue) => issue.level === 'error').length);
	let recipeWarnCount = $derived(recipeIssues.filter((issue) => issue.level !== 'error').length);
	let hasTopIssue = $derived(recipeErrorCount > 0 || recipeWarnCount > 0);
	let topIssueTitle = $derived(recipeIssues.map((issue) => issue.msg).join('\n'));
</script>

{#if !$projectLoaded}
	<div class="p-6 text-[12px]" style="color: var(--text-muted);">Open or create a project to configure conditioning.</div>
{:else}
	<div class="conditioning-page">
		<section class="conditioning-toolbar">
			<div class="toolbar-copy">
				<div class="eyebrow">Conditioning</div>
				<div class="recipe-sentence">{recipeSentence}</div>
			</div>
			<div class="toolbar-controls">
				<div class="preset-control">
					<label for="conditioning-preset">Preset</label>
					<select id="conditioning-preset" value={selectedPresetId} onchange={(e) => { pendingPresetId = e.target.value; }}>
						<option value="" disabled>Choose preset...</option>
						{#each PRESETS as preset}
							<option value={preset.id}>{preset.label}</option>
						{/each}
					</select>
					<button type="button" onclick={applySelectedPreset} disabled={!canApplySelectedPreset}>Apply</button>
				</div>
				<div class="mode-control">
					<label for="conditioning-mode">Mode</label>
					<select id="conditioning-mode" value={t.ltx2_mode || 'video'} onchange={(e) => setMode(e.target.value)}>
						<option value="video">Video</option>
						<option value="av">AV</option>
						<option value="audio">Audio</option>
					</select>
				</div>
			</div>
			<div class="top-status-strip" aria-label="Conditioning status">
				<div class="status-group objective-status" title="Detected from the current mode, generated/input targets, and active conditions. Presets are only starting points.">
					<div class="status-group-label">Objective</div>
					<div class="objective-line">
						<span class="status-dot"></span>
						<strong>{objectiveInfo.custom ? `Custom: ${objectiveInfo.label}` : objectiveInfo.label}</strong>
					</div>
				</div>
				{#if hasTopIssue}
					<div class="status-group issue-status" class:issue-status-error={recipeErrorCount > 0} class:issue-status-warn={recipeErrorCount === 0} title={topIssueTitle}>
						<div class="status-group-label">Issues</div>
						<div class="status-issue-line">
							<span class="status-dot"></span>
							<strong>{recipeErrorCount > 0 ? `${recipeErrorCount} conditioning error${recipeErrorCount === 1 ? '' : 's'}` : `${recipeWarnCount} conditioning warning${recipeWarnCount === 1 ? '' : 's'}`}</strong>
						</div>
					</div>
				{/if}
				<div class="status-group readiness-status">
					<div class="status-group-label">Status</div>
					<div class="status-lines">
						{#if readinessItems.length === 0}
							<div class="status-line status-ok" title="This recipe does not require extra dataset setup.">
								<span class="status-dot"></span>
								<span class="status-state">OK</span>
								<strong>No setup warnings</strong>
							</div>
						{:else}
							{#each readinessItems as item}
								<div class="status-line" class:status-ok={item.ok} class:status-need={!item.ok} title={item.detail}>
									<span class="status-dot"></span>
									<span class="status-state">{item.ok ? 'OK' : 'Need'}</span>
									<strong>{item.label}</strong>
								</div>
							{/each}
						{/if}
					</div>
				</div>
				<div class="status-group recipe-status">
					<div class="status-group-label">Recipe</div>
					<div class="flow-lines">
						<div class="flow-line" title={timelineSegments('video').map((seg) => `${seg.label}: ${seg.detail}`).join(', ')}>
							<span>Video</span>
							<strong>{timelineSegments('video').map((seg) => seg.label).join(' + ')}</strong>
						</div>
						{#if showAudioRow}
							<div class="flow-line" title={timelineSegments('audio').map((seg) => `${seg.label}: ${seg.detail}`).join(', ')}>
								<span>Audio</span>
								<strong>{timelineSegments('audio').map((seg) => seg.label).join(' + ')}</strong>
							</div>
						{/if}
					</div>
				</div>
			</div>
		</section>

		<section class="conditioning-builder">
			<div class="builder-panel">
				<div class="panel-title">
					<span>Pick What The Model Sees</span>
					<strong>{appliedRecipe.video.conditions.length + appliedRecipe.audio.conditions.length} active</strong>
				</div>

				{#each [['video', 'Video'], ['audio', 'Audio']] as [mod, label]}
					{@const available = mod === 'video' ? videoModeOn : audioModeOn}
					{#if available}
					<div class="modality-block">
						<div class="modality-top">
							<div>
								<strong>{label}</strong>
								<span>{recipe[mod].is_generated ? 'Train this output' : 'Use as input'}</span>
							</div>
							<div class="target-toggle">
								<button
									type="button"
									class:target-active={recipe[mod].is_generated}
									onclick={() => setGenerated(mod, true)}
									disabled={!available && mod === 'audio'}
								>
									Generate
								</button>
								<button
									type="button"
									class:target-active={!recipe[mod].is_generated}
									onclick={() => setGenerated(mod, false)}
									disabled={!available && mod === 'audio'}
								>
									Input
								</button>
							</div>
					</div>
						<div class="condition-stack">
							{#each conditionTypesFor(mod) as ty}
								{@const idx = conditionIndex(mod, ty)}
								{@const active = idx >= 0}
								{@const c = active ? recipe[mod].conditions[idx] : null}
								<div class="condition-row" class:condition-row-active={active}>
									<div
										class="condition-row-header"
										role="button"
										tabindex={available ? 0 : -1}
										onclick={() => available && toggleCondition(mod, ty)}
										onkeydown={(e) => {
											if (!available) return;
											if (e.target !== e.currentTarget) return;
											if (e.key === 'Enter' || e.key === ' ') {
												e.preventDefault();
												toggleCondition(mod, ty);
											}
										}}
									>
										<div class="condition-toggle" role="presentation" onclick={(e) => e.stopPropagation()}>
											<FormToggle label={SHORT_TYPE_LABEL[ty] || ty} checked={active} onchange={() => toggleCondition(mod, ty)} disabled={!available} tooltip={TYPE_HELP[ty]} />
										</div>
										<span class="condition-row-state">{active ? 'On' : 'Off'}</span>
									</div>
									{#if active}
										<div class="condition-param-panel">
											{#if c.type === 'extend'}
												<div class="control-grid three">
													<FormField type="number" label="Prefix frames" value={c.prefix ?? 0} oninput={(e) => setCond(mod, idx, 'prefix', Number(e.target.value))} min={0} step="1" tooltip="Leading latent frames kept clean (forward extension). 0 = off." />
													<FormField type="number" label="Suffix frames" value={c.suffix ?? 0} oninput={(e) => setCond(mod, idx, 'suffix', Number(e.target.value))} min={0} step="1" tooltip="Trailing latent frames kept clean (backward extension). 0 = off." />
													<FormField type="number" label="Probability" value={c.probability ?? ''} oninput={(e) => setCond(mod, idx, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="1.0" tooltip="Per-sample probability of applying extension." />
												</div>
												{#if $advancedMode}
													<div class="control-grid two">
														<FormField type="number" label="Prefix p" value={c.prefix_p ?? ''} oninput={(e) => setCond(mod, idx, 'prefix_p', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="shared" tooltip="Independent prefix-side probability. Blank = use the shared probability." />
														<FormField type="number" label="Suffix p" value={c.suffix_p ?? ''} oninput={(e) => setCond(mod, idx, 'suffix_p', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="shared" tooltip="Independent suffix-side probability. Blank = use the shared probability." />
													</div>
												{/if}
											{:else if c.type === 'inpaint'}
												<div class="control-grid three">
													<FormField type="number" label="Probability" value={c.probability ?? ''} oninput={(e) => setCond(mod, idx, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="0.0" tooltip="Per-sample probability of applying the mask as conditioning." />
													<FormField type="number" label="Threshold" value={c.threshold ?? 0.5} oninput={(e) => setCond(mod, idx, 'threshold', e.target.value === '' ? 0.5 : Number(e.target.value))} min={0} max={1} step="0.05" tooltip="Binarization threshold (strict >). Mask values above this are kept clean." />
													<FormToggle label="Invert" checked={c.invert} onchange={(e) => setCond(mod, idx, 'invert', e.target.checked)} tooltip="Condition the complement of the mask (generate the masked region)." />
												</div>
											{:else if c.type === 'spatial_crop'}
												<div class="control-grid two">
													<FormField type="number" label="Probability" value={c.probability ?? ''} oninput={(e) => setCond(mod, idx, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="0.0" tooltip="Per-sample probability of applying the crop region as conditioning." />
													<FormToggle label="Invert" checked={c.invert} onchange={(e) => setCond(mod, idx, 'invert', e.target.checked)} tooltip="Condition outside the rectangle instead of inside." />
												</div>
											{:else if mod === 'audio' && c.type === 'reference'}
												<p class="inspector-note">Audio reference uses dataset reference audio. No recipe parameters.</p>
											{:else}
												<div class="control-grid two">
													<FormField type="number" label={c.type === 'reference' ? 'Keep probability' : 'Probability'} value={c.probability ?? ''} oninput={(e) => setCond(mod, idx, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder={c.type === 'first_frame' ? '0.1' : c.type === 'reference' ? '1.0' : 'default'} tooltip={c.type === 'reference' ? 'Per-sample probability the reference is KEPT (values below 1.0 drop it for CFG-style training).' : 'Per-sample probability of applying first-frame conditioning.'} />
												</div>
											{/if}
											<details class="condition-details">
												<summary>What this changes</summary>
												<ul class="condition-facts">
													<li><span>Input:</span> {TYPE_RECEIVES[c.type]}</li>
													<li><span>Trains:</span> {TYPE_LEARNS[c.type]}</li>
													<li><span>Needs:</span> {TYPE_DATASET_NEEDS[c.type]}</li>
												</ul>
											</details>
										</div>
									{/if}
								</div>
							{/each}
						</div>
					</div>
					{/if}
				{/each}
			</div>
			{#if recipeIssues.length > 0}
				<div class="issue-list">
					{#each recipeIssues as issue}
						<div class="app-alert" class:app-alert-danger={issue.level === 'error'} class:app-alert-warning={issue.level !== 'error'}>
							<span class="app-alert-icon"></span>
							<span class="app-alert-body">{issue.msg}</span>
						</div>
					{/each}
				</div>
			{/if}
		</section>

		<FormGroup title="Loss">
			<div class="advanced-two-col pt-2">
				<div>
					<FormSelect label="Per-sample loss" value={recipe.per_sample_loss} options={[{ value: 'auto', label: 'Auto (enable when conditioning is active)' }, { value: 'on', label: 'On (force per-sample)' }, { value: 'off', label: 'Off (force batch-global)' }]} onchange={(e) => setPSL(e.target.value)} tooltip="Auto enables per-sample renormalization whenever conditioning is active (recommended). On/Off force it." />
				</div>
			</div>
		</FormGroup>

		{#if hasAudioRef && $advancedMode}
			<FormGroup title="Reference fine-tuning (AV)">
				<div class="advanced-two-col pt-2">
					{#if isAvIc}
						<div class="advanced-subgrid">
							<FormSelect fieldPath="training.av_cross_attention_mode" value={t.av_cross_attention_mode || 'both'} options={[{ value: 'both', label: 'both' }, { value: 'a2v_only', label: 'a2v_only' }, { value: 'v2a_only', label: 'v2a_only' }, { value: 'none', label: 'none' }]} onchange={(e) => update('av_cross_attention_mode', e.target.value)} tooltip="AV cross-modal direction control." />
							<FormToggle fieldPath="training.av_multi_ref" checked={t.av_multi_ref ?? false} onchange={(e) => update('av_multi_ref', e.target.checked)} tooltip="Mark this AV run as multi-reference (uses the plural dataset reference fields)." />
						</div>
					{/if}
					<div class="advanced-subgrid">
						<FormToggle fieldPath="training.audio_ref_use_negative_positions" checked={t.audio_ref_use_negative_positions ?? false} onchange={(e) => update('audio_ref_use_negative_positions', e.target.checked)} tooltip="Place reference-audio token positions in negative time." />
						<FormToggle fieldPath="training.audio_ref_mask_cross_attention_to_reference" checked={t.audio_ref_mask_cross_attention_to_reference ?? false} onchange={(e) => update('audio_ref_mask_cross_attention_to_reference', e.target.checked)} tooltip="Video attends only to target audio, not reference-audio tokens." />
						<FormToggle fieldPath="training.audio_ref_mask_reference_from_text_attention" checked={t.audio_ref_mask_reference_from_text_attention ?? false} onchange={(e) => update('audio_ref_mask_reference_from_text_attention', e.target.checked)} tooltip="Block reference-audio tokens from attending to text (ignored in AV IC modality-path mode)." />
					</div>
					<FormField type="number" fieldPath="training.audio_ref_identity_guidance_scale" value={t.audio_ref_identity_guidance_scale ?? 0.0} oninput={(e) => update('audio_ref_identity_guidance_scale', Number(e.target.value))} step="0.1" min={0} tooltip="Extra forward pass without reference to isolate speaker identity (0 = off, ~3.0 typical)." />
					<div class="advanced-subgrid">
						<FormToggle fieldPath="training.av_bimodal_cfg" checked={t.av_bimodal_cfg ?? false} onchange={(e) => update('av_bimodal_cfg', e.target.checked)} tooltip="Extra forward pass with cross-modal attention disabled to strengthen independent audio/video." />
						{#if t.av_bimodal_cfg}
							<FormField type="number" fieldPath="training.av_bimodal_scale" value={t.av_bimodal_scale ?? 3.0} oninput={(e) => update('av_bimodal_scale', Number(e.target.value))} step="0.1" min={1} tooltip="Bimodal guidance strength (scale-1)*delta. Default 3.0." />
						{/if}
					</div>
				</div>
			</FormGroup>
		{/if}

		<FormGroup title="Guidance (advanced, orthogonal)">
			<div class="advanced-two-col pt-2">
				<div class="advanced-card">
					<span>Graded Conditioning</span>
					<FormToggle fieldPath="training.ltx2_graded_conditioning" checked={t.ltx2_graded_conditioning ?? false} onchange={(e) => update('ltx2_graded_conditioning', e.target.checked)} tooltip="Allow explicit partial strengths for latent_idx guides and video anchors. Off keeps binary training behavior." />
				</div>
				<div class="advanced-card">
					<span>Causal Temporal Attention</span>
					<FormToggle fieldPath="training.ltx2_causal_temporal_attention" checked={t.ltx2_causal_temporal_attention ?? false} onchange={(e) => update('ltx2_causal_temporal_attention', e.target.checked)} tooltip="Restrict video self-attention to the current and earlier frames. Requires SDPA; off keeps bidirectional attention." />
				</div>
				<div class="advanced-card">
					<span>Soft AV Alignment</span>
					<FormToggle fieldPath="training.ltx2_soft_av_alignment" checked={t.ltx2_soft_av_alignment ?? false} onchange={(e) => update('ltx2_soft_av_alignment', e.target.checked)} tooltip="Bias AV cross-attention toward nearby audio/video times. Requires AV mode and SDPA." />
					{#if t.ltx2_soft_av_alignment}
						<FormField type="number" fieldPath="training.ltx2_soft_av_alignment_sigma" value={t.ltx2_soft_av_alignment_sigma ?? 1.0} oninput={(e) => update('ltx2_soft_av_alignment_sigma', Number(e.target.value))} step="0.05" min={0.01} tooltip="Gaussian temporal width in seconds. Start with 1.0 for general AV data or try 0.25 for lip-focused speech/singing." />
					{/if}
				</div>
				<div class="advanced-card">
					<span>Endpoint Keyframe Training</span>
					<FormToggle fieldPath="training.keyframe_endpoint_training" checked={t.keyframe_endpoint_training ?? false} onchange={(e) => update('keyframe_endpoint_training', e.target.checked)} tooltip="Master enable for endpoint-keyframe training." />
					{#if t.keyframe_endpoint_training}
						<div class="advanced-subgrid">
							<FormField type="number" fieldPath="training.keyframe_first_frame_p" value={t.keyframe_first_frame_p ?? 1.0} oninput={(e) => update('keyframe_first_frame_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending the first latent frame at frame_idx=0." />
							<FormField type="number" fieldPath="training.keyframe_last_frame_p" value={t.keyframe_last_frame_p ?? 1.0} oninput={(e) => update('keyframe_last_frame_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending the last latent frame." />
							<FormField type="number" fieldPath="training.keyframe_random_interior_p" value={t.keyframe_random_interior_p ?? 0.0} oninput={(e) => update('keyframe_random_interior_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending random interior latent frames." />
							<FormField type="number" fieldPath="training.keyframe_max_random_interior" value={t.keyframe_max_random_interior ?? 0} oninput={(e) => update('keyframe_max_random_interior', Number(e.target.value))} step="1" min={0} tooltip="Cap on random interior keyframes per batch." />
						</div>
					{/if}
				</div>
				<div class="advanced-card">
					<span>Video Anchor Training</span>
					<FormToggle fieldPath="training.video_anchor_training" checked={t.video_anchor_training ?? false} onchange={(e) => update('video_anchor_training', e.target.checked)} tooltip="Master enable for video-anchor training." />
					{#if t.video_anchor_training}
						<div class="advanced-subgrid">
							<FormField type="number" fieldPath="training.video_anchor_probability" value={t.video_anchor_probability ?? 0.5} oninput={(e) => update('video_anchor_probability', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of applying anchor training." />
							<FormField type="number" fieldPath="training.video_anchor_count" value={t.video_anchor_count ?? 1} oninput={(e) => update('video_anchor_count', Number(e.target.value))} step="1" min={0} tooltip="Number of random anchors per sample when random anchors are enabled." />
							<FormSelect fieldPath="training.video_anchor_strategy" value={t.video_anchor_strategy || 'endpoints_random'} options={[{ value: 'endpoints', label: 'Endpoints' }, { value: 'random', label: 'Random' }, { value: 'endpoints_random', label: 'Endpoints + Random' }]} onchange={(e) => update('video_anchor_strategy', e.target.value)} tooltip="Anchor placement strategy." />
							<FormField type="number" fieldPath="training.video_anchor_strength" value={t.video_anchor_strength ?? 1.0} oninput={(e) => update('video_anchor_strength', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="1.0 keeps clean binary anchors. Lower values require Graded Conditioning and use effective timestep (1−strength)×sigma." />
						</div>
					{/if}
				</div>
			</div>
		</FormGroup>

		<FormGroup title="External TOML Recipe">
			<div class="advanced-two-col pt-2">
				<div class="source-panel">
					<PathInput fieldPath="training.ltx2_conditioning_config" value={t.ltx2_conditioning_config || ''} oninput={(e) => update('ltx2_conditioning_config', e.target.value)} showFiles tooltip="Path to a TOML conditioning recipe with [video]/[audio] blocks. Used when the visual builder has no active recipe." />
					<div class="source-summary" class:source-summary-warn={conditioningSource.id === 'builder' && conditioningSource.externalPath}>
						<span>Source</span>
						<strong>{conditioningSource.label}</strong>
						<small>{conditioningSource.detail}</small>
					</div>
				</div>
				<div class="recipe-preview">
					<div class="recipe-preview-header">
						<span>Recipe Review</span>
						<small>Builder TOML</small>
					</div>
					<pre>{previewToml}</pre>
				</div>
			</div>
		</FormGroup>
	</div>
{/if}

<style>
	.recipe-sentence {
		padding: 0.75rem;
		color: var(--text-secondary);
		background: color-mix(in srgb, var(--bg-elevated) 78%, var(--bg-base));
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-sm);
		font-size: 12px;
		line-height: 1.55;
	}
	/* Builder board redesign */
	.conditioning-page {
		display: flex;
		flex-direction: column;
		gap: 0.9rem;
	}
	.conditioning-toolbar,
	.builder-panel {
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-sm);
	}
	.conditioning-toolbar {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		grid-template-areas:
			"copy controls"
			"status status";
		align-items: start;
		column-gap: 1rem;
		row-gap: 0.65rem;
		padding: 0.9rem 1rem;
	}
	.toolbar-copy {
		grid-area: copy;
		min-width: 0;
	}
	.toolbar-controls {
		grid-area: controls;
		display: flex;
		align-items: center;
		gap: 0.8rem;
	}
	.eyebrow,
	.panel-title span,
	.preset-control label,
	.mode-control label {
		font-family: var(--font-label);
		font-size: 10px;
		font-weight: 750;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	.conditioning-toolbar .recipe-sentence {
		margin-top: 0.3rem;
		padding: 0;
		background: transparent;
		border: 0;
		color: var(--text-primary);
		font-size: 13px;
		line-height: 1.4;
	}
	.top-status-strip {
		grid-area: status;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
		gap: 0.85rem;
		padding-top: 0.75rem;
		border-top: 1px solid var(--border-subtle);
	}
	.status-group {
		min-width: 0;
		padding: 0.45rem 0.55rem;
		background: color-mix(in srgb, var(--bg-elevated) 48%, transparent);
		border: 1px solid color-mix(in srgb, var(--border-subtle) 72%, transparent);
		border-radius: var(--radius-sm);
	}
	.status-group-label {
		margin-bottom: 0.38rem;
		font-family: var(--font-label);
		font-size: 9px;
		font-weight: 800;
		letter-spacing: 0.11em;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	.status-lines,
	.flow-lines {
		display: grid;
		gap: 0.3rem;
	}
	.status-lines {
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
	}
	.status-line,
	.flow-line {
		min-width: 0;
		color: var(--text-secondary);
		line-height: 1.15;
	}
	.status-line {
		display: grid;
		grid-template-columns: 7px 2.25rem minmax(0, 1fr);
		align-items: center;
		gap: 0.35rem;
	}
	.status-dot {
		width: 7px;
		height: 7px;
		border-radius: var(--radius-full);
		background: var(--text-muted);
	}
	.status-state,
	.flow-line span {
		font-family: var(--font-label);
		font-size: 9px;
		font-weight: 800;
		text-transform: uppercase;
		color: var(--text-muted);
	}
	.status-line strong,
	.flow-line strong {
		overflow: hidden;
		color: var(--text-primary);
		font-size: 11px;
		font-weight: 700;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.status-ok .status-dot {
		background: var(--accent);
		box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 13%, transparent);
	}
	.status-ok .status-state {
		color: var(--accent);
	}
	.status-need .status-dot {
		background: var(--warning);
		box-shadow: 0 0 0 2px color-mix(in srgb, var(--warning) 13%, transparent);
	}
	.status-need .status-state {
		color: var(--warning);
	}
	.status-issue-line {
		display: grid;
		grid-template-columns: 7px minmax(0, 1fr);
		align-items: center;
		gap: 0.45rem;
		color: var(--text-secondary);
		font-size: 11px;
		line-height: 1.2;
	}
	.status-issue-line strong {
		overflow: hidden;
		color: var(--text-primary);
		font-size: 11px;
		font-weight: 700;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.objective-line {
		display: grid;
		grid-template-columns: 7px minmax(0, 1fr);
		align-items: center;
		gap: 0.45rem;
		font-size: 11px;
		line-height: 1.2;
	}
	.objective-line .status-dot {
		background: var(--accent);
	}
	.objective-line strong {
		overflow: hidden;
		color: var(--text-primary);
		font-size: 11px;
		font-weight: 750;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.issue-status-error .status-dot {
		background: var(--danger);
	}
	.issue-status-warn .status-dot {
		background: var(--warning);
	}
	.flow-line {
		display: grid;
		grid-template-columns: 3.2rem minmax(0, 1fr);
		align-items: baseline;
		gap: 0.55rem;
		padding: 0.12rem 0;
	}
	.preset-control {
		display: grid;
		grid-template-columns: auto minmax(15rem, 1fr) auto;
		align-items: center;
		gap: 0.55rem;
	}
	.preset-control select,
	.mode-control select {
		height: 2.5rem;
		padding: 0 0.55rem;
		background: var(--bg-input);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-primary);
		font-size: 12px;
	}
	.preset-control button {
		height: 2.5rem;
		padding: 0 0.7rem;
		background: color-mix(in srgb, var(--accent) 14%, var(--bg-elevated));
		border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--border));
		border-radius: var(--radius-sm);
		color: var(--text-primary);
		font-size: 11px;
		font-weight: 700;
	}
	.preset-control button:disabled {
		cursor: default;
		opacity: 0.45;
	}
	.issue-list {
		display: grid;
		gap: 0.45rem;
		margin-top: 0.65rem;
	}
	.mode-control {
		display: grid;
		grid-template-columns: auto minmax(6.5rem, 8rem);
		align-items: center;
		gap: 0.55rem;
	}
	.conditioning-builder {
		display: block;
	}
	.builder-panel {
		padding: 1rem;
	}
	.panel-title {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.75rem;
		margin-bottom: 0.9rem;
	}
	.panel-title strong {
		color: var(--text-primary);
		font-size: 13px;
		font-weight: 750;
	}
	.modality-block {
		padding: 0.9rem;
		background: color-mix(in srgb, var(--bg-elevated) 65%, var(--bg-base));
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-sm);
	}
	.modality-block + .modality-block {
		margin-top: 0.8rem;
	}
	.modality-top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
		margin-bottom: 0.85rem;
	}
	.modality-top strong {
		color: var(--text-primary);
		font-size: 16px;
	}
	.modality-top span {
		display: block;
		margin-top: 0.1rem;
		color: var(--text-secondary);
		font-size: 12px;
	}
	.target-toggle {
		display: inline-flex;
		padding: 0.16rem;
		background: var(--bg-base);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}
	.target-toggle button {
		min-width: 5rem;
		height: 2.05rem;
		padding: 0 0.65rem;
		border-radius: calc(var(--radius-sm) - 2px);
		color: var(--text-muted);
		font-size: 11.5px;
		font-weight: 700;
	}
	.target-toggle .target-active {
		background: var(--accent);
		color: var(--bg-base);
	}
	.condition-stack {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		align-items: start;
		gap: 0.7rem;
	}
	.condition-row {
		display: grid;
		gap: 0.75rem;
		padding: 0.75rem;
		background: var(--bg-base);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		transition: background-color 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
	}
	.condition-row-active {
		background: color-mix(in srgb, var(--accent) 8%, var(--bg-base));
		border-color: color-mix(in srgb, var(--accent) 34%, var(--border));
		box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 10%, transparent);
	}
	.condition-row-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		min-height: 2.25rem;
		cursor: pointer;
	}
	.condition-row-header:focus-visible {
		outline: 2px solid color-mix(in srgb, var(--accent) 55%, transparent);
		outline-offset: 3px;
		border-radius: var(--radius-sm);
	}
	.condition-toggle {
		min-width: 13rem;
	}
	.condition-toggle :global(label) {
		align-items: center;
	}
	.condition-toggle :global(label > span) {
		color: var(--text-primary) !important;
		font-size: 14px;
		font-weight: 700;
	}
	.condition-row-state {
		color: var(--text-muted);
		font-family: var(--font-label);
		font-size: 11px;
		font-weight: 700;
		text-transform: uppercase;
	}
	.condition-row-active .condition-row-state {
		color: var(--accent);
	}
	.condition-param-panel {
		display: grid;
		gap: 0.75rem;
		padding: 0.75rem;
		background: color-mix(in srgb, var(--bg-elevated) 78%, var(--bg-base));
		border: 1px solid color-mix(in srgb, var(--accent) 18%, var(--border-subtle));
		border-radius: var(--radius-sm);
	}
	.inspector-note {
		color: var(--text-muted);
		font-size: 12px;
		line-height: 1.4;
	}
	.condition-facts {
		display: grid;
		gap: 0.28rem;
		margin: 0.45rem 0 0;
		padding: 0;
		list-style: none;
		color: var(--text-secondary);
		font-size: 11.5px;
		line-height: 1.35;
	}
	.condition-facts span {
		color: var(--text-primary);
		font-weight: 700;
	}
	.condition-details {
		padding-top: 0.25rem;
	}
	.condition-details summary {
		cursor: pointer;
		color: var(--text-muted);
		font-size: 12px;
		font-weight: 700;
	}
	.control-grid {
		display: grid;
		gap: 0.7rem;
	}
	.control-grid.two {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}
	.control-grid.three {
		grid-template-columns: repeat(3, minmax(0, 1fr));
	}
	.advanced-two-col {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.9rem;
		align-items: start;
	}
	.advanced-subgrid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.75rem;
		align-items: end;
	}
	.advanced-card {
		display: grid;
		gap: 0.75rem;
		padding: 0.85rem;
		background: var(--bg-elevated);
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-sm);
	}
	.advanced-card > span,
	.recipe-preview-header span {
		color: var(--text-secondary);
		font-family: var(--font-label);
		font-size: 12px;
		font-weight: 750;
		text-transform: uppercase;
	}
	.recipe-preview {
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-sm);
		overflow: hidden;
	}
	.source-panel {
		display: grid;
		gap: 0.65rem;
	}
	.source-summary {
		display: grid;
		gap: 0.2rem;
		padding: 0.65rem 0.75rem;
		background: color-mix(in srgb, var(--bg-elevated) 78%, var(--bg-base));
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-sm);
	}
	.source-summary-warn {
		border-color: color-mix(in srgb, var(--warning) 38%, var(--border));
		background: color-mix(in srgb, var(--warning) 8%, var(--bg-elevated));
	}
	.source-summary span {
		color: var(--text-muted);
		font-family: var(--font-label);
		font-size: 10px;
		font-weight: 800;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}
	.source-summary strong {
		color: var(--text-primary);
		font-size: 13px;
	}
	.source-summary small {
		color: var(--text-secondary);
		font-size: 11.5px;
		line-height: 1.35;
	}
	.recipe-preview-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--border-subtle);
	}
	.recipe-preview-header small {
		color: var(--text-muted);
		font-family: var(--font-label);
		font-size: 10px;
		font-weight: 700;
		text-transform: uppercase;
	}
	.recipe-preview pre {
		max-height: 12rem;
		margin: 0;
		padding: 0.75rem;
		overflow: auto;
		background: var(--console-bg);
		box-shadow: inset 0 2px 6px rgba(0,0,0,.3);
		color: var(--console-text);
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		line-height: 1.45;
		white-space: pre-wrap;
	}
	@media (max-width: 980px) {
		.conditioning-toolbar {
			grid-template-columns: 1fr;
			grid-template-areas:
				"copy"
				"controls"
				"status";
		}
		.toolbar-controls {
			display: grid;
			grid-template-columns: 1fr;
		}
		.preset-control {
			grid-template-columns: auto minmax(0, 1fr) auto;
		}
		.top-status-strip {
			grid-template-columns: 1fr;
		}
		.condition-stack {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
		.condition-facts,
		.advanced-two-col,
		.advanced-subgrid,
		.control-grid.three {
			grid-template-columns: 1fr;
		}
	}
	@media (max-width: 620px) {
		.modality-top {
			align-items: stretch;
			flex-direction: column;
		}
		.target-toggle {
			width: 100%;
		}
		.target-toggle button {
			flex: 1;
		}
		.control-grid.two {
			grid-template-columns: 1fr;
		}
		.flow-line {
			grid-template-columns: 3.2rem minmax(0, 1fr);
		}
		.condition-stack {
			grid-template-columns: 1fr;
		}
	}
</style>
