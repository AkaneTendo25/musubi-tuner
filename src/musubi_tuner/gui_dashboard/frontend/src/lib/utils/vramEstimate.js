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

function largestPixelDataset(cfg) {
	const datasets = cfg?.dataset?.datasets || [];
	if (!datasets.length) return firstDataset(cfg);
	const pixelFrames = (dataset) => {
		const frames = dataset?.type === 'image' ? 1 : Math.max(Number(dataset?.target_frames || 124), 1);
		return Math.max(Number(dataset?.resolution_w || 768), 64) * Math.max(Number(dataset?.resolution_h || 512), 64) * frames;
	};
	return datasets.reduce((largest, dataset) => pixelFrames(dataset) > pixelFrames(largest) ? dataset : largest);
}

function h3TemporalLatents(frameCount) {
	const frames = Math.max(Number(frameCount || 1), 1);
	if (frames <= 1) return { video: 1, audio: 0 };
	const alignedFrames = Math.ceil(Math.max(frames - 5, 0) / 17) * 17 + 5;
	return {
		video: Math.floor((alignedFrames - 5) / 17) * 5 + 2,
		audio: Math.floor((10 * alignedFrames + 3) / 6),
	};
}

function h3DatasetRows(cfg, dataset) {
	const t = cfg?.training || {};
	const c = cfg?.caching || {};
	const ds = dataset || {};
	const targetMode = String(ds.h3_target_mode || (ds.type === 'image' ? 'video' : 'av')).toLowerCase();
	const frames = Math.max(Number(ds.type === 'image' ? (ds.h3_image_frame_count || 1) : (ds.target_frames || 124)), 1);
	const { video: videoFrames, audio: audioFrames } = h3TemporalLatents(frames);
	const height = Math.max(Number(ds.resolution_h || 480), 32);
	const width = Math.max(Number(ds.resolution_w || 832), 32);
	const videoRows = Math.max(Math.floor(height / 32), 1) * Math.max(Math.floor(width / 32), 1) * videoFrames;
	const audioRows = 2 * audioFrames;
	let rows = 256;
	if (targetMode !== 'audio') rows += videoRows;
	if (targetMode !== 'video') rows += audioRows;

	const task = String(c.h3_task || t.h3_training_mode || 't2va').toLowerCase();
	const rowsPerVideoFrame = Math.max(Math.floor(height / 32), 1) * Math.max(Math.floor(width / 32), 1);
	if (['i2va', 'l2va'].includes(task)) rows += rowsPerVideoFrame;
	else if (task === 'fl2va') rows += 2 * rowsPerVideoFrame;
	else if (task.startsWith('ref2va')) {
		const referenceFrames = Math.max(Number(ds.reference_frames || frames), 1);
		const refLatents = h3TemporalLatents(referenceFrames);
		rows += rowsPerVideoFrame * refLatents.video;
		if (targetMode !== 'video' && (ds.control_audio_directory || ds.control_video_directory)) {
			rows += 2 * refLatents.audio;
		}
	}

	const anchors = String(t.h3_keyframe_anchors || '').split(',').filter((value) => value.trim()).length;
	const randomAnchors = Math.max(Number(t.h3_keyframe_random_count || 0), 0);
	rows += Math.max(anchors, randomAnchors) * rowsPerVideoFrame;
	rows += Math.max(Number(t.h3_extension_video_frames || 0), 0) * rowsPerVideoFrame;
	rows += 2 * Math.max(Number(t.h3_extension_audio_latents || 0), 0);
	return rows;
}

function heaviestH3Dataset(cfg) {
	const datasets = cfg?.dataset?.datasets || [];
	if (!datasets.length) return firstDataset(cfg);
	return datasets.reduce((heaviest, dataset) =>
		h3DatasetRows(cfg, dataset) > h3DatasetRows(cfg, heaviest) ? dataset : heaviest
	);
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
	const ds = largestPixelDataset(cfg);
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
	const ds = heaviestH3Dataset(cfg);
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
	const isAutomagic3 = optType === 'automagic3' || optType === 'automagicv3';
	const is8bitOpt = optType.includes('8bit');
	const isScheduleFree = optType.includes('schedulefree') || optType === 'automagic';
	const optBytesPerParam = isAutomagic3 ? 1 : (is8bitOpt ? 6 : (isScheduleFree ? 14 : 12));
	const optimStates = (loraParamCount * optBytesPerParam) / (1024 ** 3);
	const loraGrads = loraParamsGB;

	const resolutionW = Math.max(Number(ds.resolution_w || 768), 64);
	const resolutionH = Math.max(Number(ds.resolution_h || 512), 64);
	// H3 processes dataset batches item-by-item and accumulates their gradients,
	// so batch size raises step time but does not multiply the live activation graph.
	const memoryBatchSize = 1;
	const sourceFrames = Math.max(Number(ds.type === 'image' ? (ds.h3_image_frame_count || 1) : (ds.target_frames || 124)), 1);
	const { video: latentFrames } = h3TemporalLatents(sourceFrames);
	const latentHeight = Math.max(1, Math.floor(resolutionH / 32));
	const latentWidth = Math.max(1, Math.floor(resolutionW / 32));
	const seqTokens = h3DatasetRows(cfg, ds);

	const hiddenDim = 5376;
	const bytesPerValue = isW8A8 ? 1 : 2;
	const standardBlocks = usesCheckpointing ? totalBlocks - checkpointedBlocks : totalBlocks;

	let activationUnits;
	if (!usesCheckpointing) activationUnits = totalBlocks * 10;
	else if (blockwise) activationUnits = Math.max(2, standardBlocks + 2);
	else activationUnits = checkpointedBlocks * 2 + standardBlocks * 10;
	let activations = (memoryBatchSize * seqTokens * hiddenDim * bytesPerValue * activationUnits) / (1024 ** 3);

	if ((t.ffn_chunk_size || 0) > 0) activations *= 0.90;
	if (t.gradient_checkpointing_cpu_offload && usesCheckpointing) activations *= 0.35;

	const latentBytes = memoryBatchSize * 128 * latentFrames * latentHeight * latentWidth * 2 * 2;
	const textEmbedBytes = memoryBatchSize * 256 * 7680 * 2;
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
