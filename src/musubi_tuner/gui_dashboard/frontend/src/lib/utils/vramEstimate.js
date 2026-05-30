function gemmaSize(cfg) {
	if (cfg?.gemma_load_in_4bit) return 6;
	if (cfg?.gemma_load_in_8bit) return 12;
	return 24;
}

function vaeSize(cfg) {
	const dtype = cfg?.vae_dtype || 'bfloat16';
	return dtype === 'float32' ? 3.0 : 1.5;
}

function firstDataset(cfg) {
	const allDatasets = cfg?.dataset?.datasets || [];
	return allDatasets.find((d) => d?.type === 'video' || d?.type === 'image') || allDatasets[0] || {};
}

function blocksToCheckpointValue(t) {
	const value = t?.blocks_to_checkpoint;
	if (value === '' || value === null || value === undefined) return -1;
	const numberValue = Number(value);
	return Number.isFinite(numberValue) ? numberValue : -1;
}

function checkpointingActive(t) {
	return blocksToCheckpointValue(t) !== 0 && (!!t?.blockwise_checkpointing || t?.gradient_checkpointing !== false);
}

export function estimateLatentCaching(cfg) {
	if (!cfg?.caching) return null;
	const c = cfg.caching;
	const vae = vaeSize(c);
	const ds = firstDataset(cfg);
	const resW = Math.max(Number(ds.resolution_w || 768), 64);
	const resH = Math.max(Number(ds.resolution_h || 512), 64);
	const frames = Math.max(Number(ds.target_frames || 33), 1);
	const basePixelFrames = 512 * 768 * 33;
	const pixelFrames = resW * resH * frames;
	const resScale = Math.max(0.5, Math.min(pixelFrames / basePixelFrames, 4.0));
	const hasSpatialTiling = !!(c.vae_spatial_tile_size || c.vae_chunk_size);
	const hasTemporalTiling = !!c.vae_temporal_tile_size;
	const tilingFactor = (hasSpatialTiling && hasTemporalTiling) ? 0.2 :
		hasSpatialTiling ? 0.3 : hasTemporalTiling ? 0.5 : 1.0;
	const buffer = 2.5 * resScale * tilingFactor;
	return {
		total: Math.max(vae + buffer, 1),
		parts: [
			{ label: 'VAE', value: vae, color: 'var(--accent)' },
			{ label: 'Activations', value: buffer, color: 'var(--info)' },
		]
	};
}

export function estimateTextCaching(cfg) {
	if (!cfg?.caching) return null;
	const c = cfg.caching;
	const gemma = gemmaSize(c);
	const buffer = 2.0;
	return {
		total: Math.max(gemma + buffer, 1),
		parts: [
			{ label: 'Gemma', value: gemma, color: 'var(--accent)' },
			{ label: 'Buffer', value: buffer, color: 'var(--info)' },
		]
	};
}

export function estimateTraining(cfg) {
	if (!cfg?.training) return null;
	const t = cfg.training;
	const ds = firstDataset(cfg);

	const ltxVersion = String(t.ltx_version || '2.3');
	const ditBF16 = ltxVersion === '2.3' ? 42 : 39;
	const isFp8 = !!t.fp8_base;
	const isW8A8 = !!t.fp8_w8a8;
	const isNF4 = !!t.nf4_base;
	let ditBase = isNF4 ? (ditBF16 / 4) : isFp8 ? (ditBF16 / 2) : ditBF16;

	const totalBlocks = 48;
	const blocksToCheckpoint = blocksToCheckpointValue(t);
	const blockwise = !!t.blockwise_checkpointing && blocksToCheckpoint !== 0;
	const checkpointedBlocks = blocksToCheckpoint === 0
		? 0
		: blocksToCheckpoint < 0
			? totalBlocks
			: Math.min(Math.max(blocksToCheckpoint, 0), totalBlocks);
	const blocksToSwap = Math.min(Math.max(Number(t.blocks_to_swap || 0), 0), totalBlocks - 1);
	const blockSize = ditBase / totalBlocks;
	const swapSavings = blocksToSwap * blockSize * 0.95;
	const residentBlocksAfterSwap = Math.max(totalBlocks - (swapSavings / Math.max(blockSize, 0.0001)), 0);
	const blockwiseWeightSavings = blockwise
		? Math.min(checkpointedBlocks, residentBlocksAfterSwap) * blockSize * 0.80
		: 0;
	const dit = Math.max(ditBase - swapSavings - blockwiseWeightSavings, 1.0);

	const rank = Math.max(Number(t.network_dim || 16), 1);
	const mode = String(t.ltx2_mode || 'video');
	const isAV = mode === 'av';
	const loraBasePerRank = isAV ? 12.75 / 1024 : 6.0 / 1024;
	const presetMultiplier = {
		t2v: 1.0,
		v2v: 1.44,
		video_sa: 0.37,
		video_sa_ff: 0.56,
		video_sa_ca_ff: 0.74,
		audio: 0.37,
		audio_v2a: 0.52,
		audio_ref_ic: 0.63,
		av_ic: 1.44,
		video_ref_only_av: 1.44,
		full: 2.1
	}[t.lora_target_preset] || 1.0;
	const loraParamsGB = rank * loraBasePerRank * presetMultiplier;

	const loraParamCount = loraParamsGB * (1024 ** 3) / 2;
	const optType = String(t.optimizer_type || 'adamw8bit').toLowerCase();
	const is8bitOpt = optType.includes('8bit');
	const isScheduleFree = optType.includes('schedulefree') || optType === 'automagic';
	const optBytesPerParam = is8bitOpt ? 6 : (isScheduleFree ? 14 : 12);
	const optimStates = (loraParamCount * optBytesPerParam) / (1024 ** 3);
	const loraGrads = loraParamsGB;

	const resolutionW = Math.max(Number(ds.resolution_w || 768), 64);
	const resolutionH = Math.max(Number(ds.resolution_h || 512), 64);
	const sourceFrames = Math.max(Number(ds.target_frames || 33), 1);
	const batchSize = Math.max(Number(ds.batch_size || 1), 1);

	const latentFrames = Math.max(1, Math.floor((sourceFrames - 1) / 8) + 1);
	const latentHeight = Math.max(1, Math.floor(resolutionH / 32));
	const latentWidth = Math.max(1, Math.floor(resolutionW / 32));
	let seqTokens = latentFrames * latentHeight * latentWidth;
	if (mode === 'audio') seqTokens = Math.round(sourceFrames);

	const hiddenDim = mode === 'audio' ? 2048 : 4096;
	const bytesPerValue = isW8A8 ? 1 : 2;
	const usesCheckpointing = checkpointingActive(t);
	const standardBlocks = usesCheckpointing ? totalBlocks - checkpointedBlocks : totalBlocks;

	let activationUnits;
	if (!usesCheckpointing) activationUnits = totalBlocks * 10;
	else if (blockwise) activationUnits = Math.max(2, standardBlocks + 2);
	else activationUnits = totalBlocks * 2;
	let activations = (batchSize * seqTokens * hiddenDim * bytesPerValue * activationUnits) / (1024 ** 3);

	if (isAV) activations *= 1.25;
	if ((t.ffn_chunk_size || 0) > 0) activations *= 0.90;
	if (t.split_attn_mode || t.split_attn_target) activations *= 0.92;
	if (t.gradient_checkpointing_cpu_offload && usesCheckpointing) activations *= 0.35;

	const latentBytes = batchSize * 128 * latentFrames * latentHeight * latentWidth * 2 * 2;
	const textEmbedBytes = batchSize * 256 * (isAV ? 7680 : 3840) * 2;
	const bufferGB = (latentBytes + textEmbedBytes) / (1024 ** 3);
	let activationBuffers = 0.5 + bufferGB;
	if (t.img_in_txt_in_offloading) activationBuffers = Math.max(0.2, activationBuffers - 0.3);

	const activationTotal = Math.max(0.3, activations + activationBuffers);
	const gradAccum = Math.max(Number(t.gradient_accumulation_steps || 1), 1);
	const gradAccumOverhead = gradAccum > 1 ? loraGrads * 0.4 : 0;

	let preservationOverhead = 0;
	if (t.blank_preservation) preservationOverhead += activationTotal * 0.35;
	if (t.dop) preservationOverhead += activationTotal * 0.35;
	if (t.audio_dop) preservationOverhead += activationTotal * 0.35;
	if (t.prior_divergence) preservationOverhead += activationTotal * 0.15;

	let selfFlowOverhead = 0;
	if (t.self_flow) {
		const teacherMode = String(t.self_flow_teacher_mode || 'base').toLowerCase();
		if (teacherMode === 'ema') selfFlowOverhead += loraParamsGB;
		else if (teacherMode === 'partial_ema') selfFlowOverhead += Math.max(loraParamsGB / totalBlocks, 0.01);

		const hasAudioProjector = (mode === 'av' || mode === 'audio') && Number(t.self_flow_lambda_audio || 0) > 0;
		selfFlowOverhead += hasAudioProjector ? 0.03 : 0.02;
		selfFlowOverhead += activationTotal * (t.self_flow_offload_teacher_features ? 0.03 : 0.10);
	}

	const crepaOverhead = t.crepa ? (String(t.crepa_mode || 'backbone') === 'dino' ? 0.08 : 0.15) : 0;
	const total = dit + loraParamsGB + optimStates + loraGrads + activationTotal + gradAccumOverhead + preservationOverhead + selfFlowOverhead + crepaOverhead;

	const hasSamplePrompts = !!(t.sample_prompts || t.sample_prompts_text);
	const samplingSpike = hasSamplePrompts && !!(t.sample_at_first || t.sample_every_n_steps || t.sample_every_n_epochs);
	const preservationGemmaSpike = !!(t.blank_preservation || t.dop || t.audio_dop) && !t.use_precached_preservation;

	const parts = [
		{ label: 'DiT', value: dit, color: 'var(--accent)' },
		{ label: 'LoRA', value: loraParamsGB, color: 'var(--warning)' },
		{ label: 'Optimizer', value: optimStates, color: 'var(--warning)' },
		{ label: 'Grads', value: loraGrads, color: 'var(--info)' },
		{ label: 'Activ.', value: activationTotal, color: 'var(--success)' },
	];
	if (gradAccumOverhead > 0) parts.push({ label: 'GradAccum', value: gradAccumOverhead, color: 'var(--info)' });
	if (preservationOverhead > 0) parts.push({ label: 'Preserv.', value: preservationOverhead, color: 'var(--danger)' });
	if (selfFlowOverhead > 0) parts.push({ label: 'Self-Flow', value: selfFlowOverhead, color: 'var(--danger)' });
	if (crepaOverhead > 0) parts.push({ label: 'CREPA', value: crepaOverhead, color: 'var(--secondary, var(--info))' });

	return {
		total: Math.max(total, 2),
		parts,
		swap: swapSavings,
		blockwiseSavings: blockwiseWeightSavings,
		noGemma: true,
		samplingSpike,
		preservationGemmaSpike,
		blockwise,
	};
}
