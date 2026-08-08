export function emptyConditioningModality() {
	return { is_generated: true, conditions: [] };
}

export function emptyConditioningRecipe() {
	return {
		enabled: false,
		per_sample_loss: 'auto',
		video: emptyConditioningModality(),
		audio: emptyConditioningModality(),
	};
}

export function normalizeConditioningRecipe(configOrTraining) {
	const t = configOrTraining?.training || configOrTraining || {};
	const r = t.conditioning_recipe;
	if (!r || typeof r !== 'object') return emptyConditioningRecipe();
	return {
		enabled: !!r.enabled,
		per_sample_loss: r.per_sample_loss || 'auto',
		video: {
			is_generated: r.video?.is_generated !== false,
			conditions: Array.isArray(r.video?.conditions) ? r.video.conditions : [],
		},
		audio: {
			is_generated: r.audio?.is_generated !== false,
			conditions: Array.isArray(r.audio?.conditions) ? r.audio.conditions : [],
		},
	};
}

export function getModeEffectiveConditioningRecipe(configOrTraining) {
	const t = configOrTraining?.training || configOrTraining || {};
	const recipe = normalizeConditioningRecipe(t);
	const mode = t.ltx2_mode || 'video';
	if (mode === 'video') {
		recipe.audio = emptyConditioningModality();
	} else if (mode === 'audio') {
		recipe.video = emptyConditioningModality();
	}
	recipe.enabled = recipeHasActive(recipe);
	return recipe;
}

export function recipeHasActive(recipe) {
	return !!(recipe?.enabled && (
		(recipe.video?.conditions || []).length > 0 ||
		(recipe.audio?.conditions || []).length > 0 ||
		recipe.video?.is_generated === false ||
		recipe.audio?.is_generated === false
	));
}

export function modalityHasRecipeContent(modality) {
	return !!((modality?.conditions || []).length > 0 || modality?.is_generated === false);
}

function hasCondition(recipe, modality, type) {
	return (recipe?.[modality]?.conditions || []).some((c) => c.type === type);
}

function finiteNumber(value) {
	if (value === '' || value === null || value === undefined) return null;
	const n = Number(value);
	return Number.isFinite(n) ? n : NaN;
}

function pushRangeIssue(issues, value, fieldName, label) {
	if (value === null) return;
	if (!Number.isFinite(value) || value < 0 || value > 1) {
		issues.push({ level: 'error', msg: `${label} must be between 0 and 1.`, field: fieldName });
	}
}

export function getConditioningSource(config) {
	const t = config?.training || {};
	const recipe = getModeEffectiveConditioningRecipe(t);
	const builderActive = recipeHasActive(recipe);
	const externalPath = (t.ltx2_conditioning_config || '').trim();
	if (builderActive) {
		return {
			id: 'builder',
			label: 'Visual builder',
			detail: externalPath ? 'External TOML is set but ignored while the visual builder has an active recipe.' : 'The visual builder recipe will be exported to TOML at launch.',
			builderActive,
			externalPath,
		};
	}
	if (externalPath) {
		return {
			id: 'external',
			label: 'External TOML',
			detail: 'The external TOML recipe path will be passed through at launch.',
			builderActive,
			externalPath,
		};
	}
	return {
		id: 'legacy',
		label: 'Legacy/default flags',
		detail: 'No visual recipe or external TOML is active.',
		builderActive,
		externalPath,
	};
}

export function getConditioningIssues(config) {
	const t = config?.training || {};
	const r = getModeEffectiveConditioningRecipe(t);
	const issues = [];
	const active = recipeHasActive(r);
	const mode = t.ltx2_mode || 'video';
	const audioModeOn = ['av', 'audio'].includes(mode);
	const videoModeOn = ['av', 'video'].includes(mode);
	const vFrozen = !r.video.is_generated;
	const aFrozen = !r.audio.is_generated;
	const hasVideoRef = hasCondition(r, 'video', 'reference');
	const hasAudioRef = hasCondition(r, 'audio', 'reference');
	const anyRef = hasVideoRef || hasAudioRef;

	if (active) {
		const vConflict = r.video.conditions.filter((c) => ['spatial_crop', 'inpaint', 'extend'].includes(c.type));
		if (vFrozen && aFrozen)
			issues.push({ level: 'error', msg: 'Both modalities are frozen - at least one must be Generated.' });
		if (vFrozen && vConflict.length)
			issues.push({ level: 'error', msg: 'Video is frozen (v2a) but has spatial_crop / mask / prefix-suffix conditions, which it cannot combine with. Remove them, or set Video to Generate.' });
		if (anyRef && (vFrozen || aFrozen))
			issues.push({ level: 'error', msg: 'A reference condition cannot be combined with a frozen modality (directional). Remove the reference, or Generate both modalities.' });
		if ((vFrozen || aFrozen) && mode !== 'av')
			issues.push({ level: 'error', msg: 'Directional training (a frozen modality) requires Mode = AV. Set Mode to AV above, or Generate both modalities.' });
		if (vFrozen && (t.video_anchor_training || t.hfato))
			issues.push({ level: 'error', msg: 'Video is frozen (v2a) but video-anchor training / HFATO is enabled - they modify the video the recipe keeps clean. Disable them.' });

		for (const [name, mod] of [['video', r.video], ['audio', r.audio]]) {
			for (const c of mod.conditions) {
				const prefix = Number(c.prefix || 0);
				const suffix = Number(c.suffix || 0);
				if (c.type === 'extend' && prefix <= 0 && suffix <= 0)
					issues.push({ level: 'error', msg: `The ${name} prefix/suffix condition needs prefix or suffix greater than 0.` });
				if (c.type === 'extend' && (prefix < 0 || suffix < 0))
					issues.push({ level: 'error', msg: `The ${name} prefix / suffix cannot be negative.` });
				if (c.probability === 0)
					issues.push({ level: 'warn', msg: `The ${name} ${c.type} condition has probability 0 - it will never apply.` });
				for (const key of ['probability', 'prefix_p', 'suffix_p']) {
					const val = finiteNumber(c[key]);
					pushRangeIssue(issues, val, null, `The ${name} ${c.type} ${key}`);
				}
				const threshold = finiteNumber(c.threshold);
				if (c.type === 'inpaint' && threshold !== null && (!Number.isFinite(threshold) || threshold < 0 || threshold > 1))
					issues.push({ level: 'error', msg: `The ${name} mask threshold must be between 0 and 1.` });
			}
		}
		if (!hasCondition(r, 'video', 'first_frame') && !vFrozen && !aFrozen && !anyRef)
			issues.push({ level: 'warn', msg: 'This recipe turns OFF first-frame (I2V) conditioning that was on by default. Add a First frame condition to keep it.' });
	}

	if (t.keyframe_endpoint_training) {
		pushRangeIssue(issues, finiteNumber(t.keyframe_first_frame_p ?? 1.0), 'training.keyframe_first_frame_p', 'Keyframe first-frame probability');
		pushRangeIssue(issues, finiteNumber(t.keyframe_last_frame_p ?? 1.0), 'training.keyframe_last_frame_p', 'Keyframe last-frame probability');
		pushRangeIssue(issues, finiteNumber(t.keyframe_random_interior_p ?? 0.0), 'training.keyframe_random_interior_p', 'Keyframe random-interior probability');
		const maxInterior = finiteNumber(t.keyframe_max_random_interior ?? 0);
		if (!Number.isFinite(maxInterior) || maxInterior < 0) {
			issues.push({ level: 'error', msg: 'Keyframe max random interior must be at least 0.', field: 'training.keyframe_max_random_interior' });
		}
		const first = finiteNumber(t.keyframe_first_frame_p ?? 1.0);
		const last = finiteNumber(t.keyframe_last_frame_p ?? 1.0);
		const interior = finiteNumber(t.keyframe_random_interior_p ?? 0.0);
		if (first === 0 && last === 0 && interior === 0) {
			issues.push({ level: 'warn', msg: 'Endpoint keyframe training is enabled, but all keyframe probabilities are 0.' });
		}
	}

	return {
		errors: issues.filter((issue) => issue.level === 'error'),
		warnings: issues.filter((issue) => issue.level !== 'error'),
		all: issues,
	};
}

export function detectConditioningObjective(configOrTraining) {
	const t = configOrTraining?.training || configOrTraining || {};
	const recipe = getModeEffectiveConditioningRecipe(t);
	const mode = t.ltx2_mode || 'video';
	const videoGenerated = recipe.video.is_generated;
	const audioGenerated = recipe.audio.is_generated;
	const videoHas = (type) => hasCondition(recipe, 'video', type);
	const audioHas = (type) => hasCondition(recipe, 'audio', type);
	let id = 'custom';
	let label = 'Custom objective';

	if (mode === 'video') {
		if (videoHas('reference')) [id, label] = ['v2v_ic', 'Reference video'];
		else if (videoHas('inpaint')) [id, label] = ['video_inpainting', 'Video inpainting'];
		else if (videoHas('spatial_crop')) [id, label] = ['video_outpainting', 'Video outpainting'];
		else if (videoHas('extend')) [id, label] = ['video_extension', 'Video extension'];
		else if (videoHas('first_frame')) [id, label] = ['i2v', 'Image-to-video'];
		else [id, label] = ['t2v', 'Text-to-video'];
	} else if (mode === 'audio') {
		if (audioHas('reference')) [id, label] = ['a2a', 'Reference audio'];
		else if (audioHas('inpaint')) [id, label] = ['audio_inpainting', 'Audio inpainting'];
		else if (audioHas('extend')) [id, label] = ['audio_extension', 'Audio extension'];
		else [id, label] = ['t2a', 'Text-to-audio'];
	} else if (!videoGenerated && audioGenerated) {
		if (audioHas('inpaint')) [id, label] = ['custom', 'Audio inpainting from video'];
		else if (audioHas('extend')) [id, label] = ['custom', 'Audio extension from video'];
		else if (audioHas('reference')) [id, label] = ['custom', 'Reference audio from video'];
		else [id, label] = ['v2a', 'Video-to-audio'];
	} else if (videoGenerated && !audioGenerated) {
		if (videoHas('inpaint')) [id, label] = ['custom', 'Video inpainting from audio'];
		else if (videoHas('spatial_crop')) [id, label] = ['custom', 'Video outpainting from audio'];
		else if (videoHas('extend')) [id, label] = ['custom', 'Video extension from audio'];
		else if (videoHas('reference')) [id, label] = ['custom', 'Reference video from audio'];
		else [id, label] = ['a2v', 'Audio-to-video'];
	} else if (videoHas('reference') && audioHas('reference')) {
		[id, label] = ['av2av_ic', 'AV reference training'];
	} else if (videoGenerated && audioGenerated) {
		if (videoHas('reference')) [id, label] = ['custom', 'Reference video + audio generation'];
		else if (audioHas('reference')) [id, label] = ['custom', 'Video generation + reference audio'];
		else if (videoHas('inpaint')) [id, label] = ['custom', 'Video inpainting + audio generation'];
		else if (videoHas('spatial_crop')) [id, label] = ['custom', 'Video outpainting + audio generation'];
		else if (audioHas('inpaint')) [id, label] = ['custom', 'Video generation + audio inpainting'];
		else if (audioHas('extend')) [id, label] = ['custom', 'Video generation + audio extension'];
		else if (videoHas('first_frame')) [id, label] = ['i2v', 'Image-to-video/audio'];
		else if (videoHas('extend')) [id, label] = ['video_extension', 'Video extension'];
		else [id, label] = ['t2v', 'Text-to-video/audio'];
	}

	return { id, label, custom: id === 'custom' };
}
