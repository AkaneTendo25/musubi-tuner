function h3QwenSize(cfg) {
	// H3 uses the 32B Qwen3-VL conditioner truncated after raw layer 50.
	// Values are resident checkpoint weights in GiB. BF16, INT8 ConvRot, and
	// NVFP4/AWQ come from the official Comfy-Org/MiniMax-H3 file sizes.
	switch (cfg?.h3_text_encoder_quantization || 'none') {
		case 'nvfp4_awq': return 14.61;
		case 'nf4': return 14.0;
		case 'int8': return 25.28;
		default: return 47.97;
	}
}

function vaeSize(cfg) {
	const dtype = cfg?.vae_dtype || 'bfloat16';
	return dtype === 'float32' ? 3.0 : 1.5;
}

function firstDataset(cfg) {
	const allDatasets = cfg?.dataset?.datasets || [];
	return allDatasets.find((d) => d?.type === 'video' || d?.type === 'image') || allDatasets[0] || {};
}

function h3BaseSize(t) {
	// H3 has 33.1B parameters, including 13.0B in the 50 AdaLN projections.
	// A rank-16 AdaLN replacement has about 77M parameters and scales linearly with rank.
	const rank = Number(t?.h3_adaln_rank);
	const hasReducedAdaln = Number.isFinite(rank) && rank > 0;
	const convRotBillions = 33.1 - 13.0;
	const adaLnBillions = hasReducedAdaln ? (0.077 * rank / 16) : 13.0;
	const paramsBillions = convRotBillions + adaLnBillions;
	const gibPerBillionBytes = 1e9 / (1024 ** 3);

	// The released pruned ConvRot checkpoint is 19.53 GiB, including quantization metadata.
	if (t?.int8_convrot_base) return 19.53;
	if (t?.h3_convrot_int8) {
		// Online ConvRot INT8 does not quantize the AdaLN projections. Without
		// low-rank AdaLN those 13B parameters remain BF16 and must stay in the estimate.
		return (convRotBillions * 1.04 + adaLnBillions * 2) * gibPerBillionBytes;
	}
	if (t?.fp8_base || t?.fp8_scaled) {
		return paramsBillions * gibPerBillionBytes * 1.04;
	}
	return paramsBillions * gibPerBillionBytes * 2;
}

export function estimateLatentCaching(cfg) {
	if (!cfg?.caching || cfg.caching.model_type !== 'minimax_h3') return null;
	const c = cfg.caching;
	const videoVae = vaeSize(c);
	const audioVae = c.h3_audio_vae ? 0.8 : 0;
	const vae = videoVae + audioVae;
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
	// Decoder-free encoding still retains convolution workspaces. This baseline
	// is calibrated for H3's 4x temporal and 16x spatial VAE contract.
	const buffer = 4.8 * resScale * tilingFactor;
	return {
		total: Math.max(vae + buffer, 1),
		confidence: 'medium',
		basis: 'H3 VAE tensor-shape model',
		parts: [
			{ label: 'VAE', value: vae, color: 'var(--accent)' },
			{ label: 'Activations', value: buffer, color: 'var(--info)' },
		]
	};
}

export function estimateTextCaching(cfg) {
	if (!cfg?.caching || cfg.caching.model_type !== 'minimax_h3') return null;
	const c = cfg.caching;
	const qwen = h3QwenSize(c);
	const buffer = c.h3_text_encoder_quantization === 'none' ? 4.0 : 3.0;
	return {
		total: Math.max(qwen + buffer, 1),
		confidence: 'medium',
		basis: '32B Qwen3-VL checkpoint size + encode workspace',
		parts: [
			{ label: 'Qwen3-VL 32B', value: qwen, color: 'var(--accent)' },
			{ label: 'Buffer', value: buffer, color: 'var(--info)' },
		]
	};
}

export function estimateTraining(cfg) {
	if (!cfg?.training || cfg.training.model_type !== 'minimax_h3') return null;
	const t = cfg.training;
	const ds = firstDataset(cfg);
	const isW8A8 = !!t.fp8_w8a8;
	let ditBase = h3BaseSize(t);

	const totalBlocks = 50;
	const configuredCheckpointBlocks = t.h3_gradient_checkpointing_blocks;
	const blocksToCheckpoint = configuredCheckpointBlocks === '' || configuredCheckpointBlocks === null || configuredCheckpointBlocks === undefined
		? -1
		: Number(configuredCheckpointBlocks);
	const usesCheckpointing = !!t.gradient_checkpointing;
	const blockwise = false;
	const checkpointedBlocks = !usesCheckpointing || blocksToCheckpoint === 0
		? 0
		: !Number.isFinite(blocksToCheckpoint) || blocksToCheckpoint < 0
			? totalBlocks
			: Math.min(Math.max(blocksToCheckpoint, 0), totalBlocks);
	const maxBlocksToSwap = t.block_swap_h2d_only && t.block_swap_granularity === 'layer'
		? totalBlocks
		: totalBlocks - 2;
	const blocksToSwap = Math.min(Math.max(Number(t.blocks_to_swap || 0), 0), maxBlocksToSwap);
	const blockSize = ditBase / totalBlocks;
	const swapSavings = blocksToSwap * blockSize * 0.95;
	const residentBlocksAfterSwap = Math.max(totalBlocks - (swapSavings / Math.max(blockSize, 0.0001)), 0);
	const blockwiseWeightSavings = blockwise
		? Math.min(checkpointedBlocks, residentBlocksAfterSwap) * blockSize * 0.80
		: 0;
	const dit = Math.max(ditBase - swapSavings - blockwiseWeightSavings, 1.0);

	const rank = Math.max(Number(t.network_dim || 16), 1);
	const mode = 'av';
	const isAV = true;
	const loraBasePerRank = 12.75 / 1024;
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

	const latentFrames = Math.max(1, Math.floor((sourceFrames - 1) / 4) + 1);
	const latentHeight = Math.max(1, Math.floor(resolutionH / 32));
	const latentWidth = Math.max(1, Math.floor(resolutionW / 32));
	let seqTokens = latentFrames * latentHeight * latentWidth;
	if (mode === 'audio') seqTokens = Math.round(sourceFrames);

	const hiddenDim = 5376;
	const bytesPerValue = isW8A8 ? 1 : 2;
	const standardBlocks = usesCheckpointing ? totalBlocks - checkpointedBlocks : totalBlocks;

	let activationUnits;
	if (!usesCheckpointing) activationUnits = totalBlocks * 10;
	else if (blockwise) activationUnits = Math.max(2, standardBlocks + 2);
	else activationUnits = checkpointedBlocks * 2 + standardBlocks * 10;
	let activations = (batchSize * seqTokens * hiddenDim * bytesPerValue * activationUnits) / (1024 ** 3);

	if (isAV) activations *= 1.25;
	if ((t.ffn_chunk_size || 0) > 0) activations *= 0.90;
	if (t.gradient_checkpointing_cpu_offload && usesCheckpointing) activations *= 0.35;

	const latentBytes = batchSize * 128 * latentFrames * latentHeight * latentWidth * 2 * 2;
	const textEmbedBytes = batchSize * 256 * (isAV ? 7680 : 3840) * 2;
	const bufferGB = (latentBytes + textEmbedBytes) / (1024 ** 3);
	// CUDA workspaces and allocator fragmentation contribute about 2.5 GiB on
	// the calibrated 832x480x124 BF16 LoRA run (14.45 GiB allocated peak).
	let activationBuffers = 2.5 + bufferGB;
	if (t.img_in_txt_in_offloading) activationBuffers = Math.max(2.2, activationBuffers - 0.3);

	const activationTotal = Math.max(0.3, activations + activationBuffers);
	const gradAccum = Math.max(Number(t.gradient_accumulation_steps || 1), 1);
	const gradAccumOverhead = gradAccum > 1 ? loraGrads * 0.4 : 0;

	// The frozen-base pass is sequential/no-grad. Paired workbox measurements
	// showed the same 14.45 GiB allocated peak with preservation on and off.
	const preservationOverhead = 0;
	// The empty-prompt teacher is also sequential/no-grad and reuses the same
	// working-memory high-water mark as preservation.
	const guidanceOverhead = 0;
	const crepaOverhead = t.crepa ? 0.15 : 0;
	const total = dit + loraParamsGB + optimStates + loraGrads + activationTotal + gradAccumOverhead + preservationOverhead + guidanceOverhead + crepaOverhead;

	const hasSamplePrompts = !!(t.sample_prompts || t.sample_prompts_text);
	const samplingSpike = hasSamplePrompts && !!(t.sample_at_first || t.sample_every_n_steps || t.sample_every_n_epochs);

	const parts = [
		{ label: 'DiT', value: dit, color: 'var(--accent)' },
		{ label: 'LoRA', value: loraParamsGB, color: 'var(--warning)' },
		{ label: 'Optimizer', value: optimStates, color: 'var(--warning)' },
		{ label: 'Grads', value: loraGrads, color: 'var(--info)' },
		{ label: 'Activ.', value: activationTotal, color: 'var(--success)' },
	];
	if (gradAccumOverhead > 0) parts.push({ label: 'GradAccum', value: gradAccumOverhead, color: 'var(--info)' });
	if (preservationOverhead > 0) parts.push({ label: 'Preserv.', value: preservationOverhead, color: 'var(--danger)' });
	if (guidanceOverhead > 0) parts.push({ label: 'Guidance', value: guidanceOverhead, color: 'var(--danger)' });
	if (crepaOverhead > 0) parts.push({ label: 'CREPA', value: crepaOverhead, color: 'var(--secondary, var(--info))' });

	return {
		total: Math.max(total, 2),
		parts,
		swap: swapSavings,
		blockwiseSavings: blockwiseWeightSavings,
		confidence: 'medium',
		basis: 'H3 parameter residency + calibrated tensor-shape activation model',
		samplingSpike,
		blockwise,
	};
}
