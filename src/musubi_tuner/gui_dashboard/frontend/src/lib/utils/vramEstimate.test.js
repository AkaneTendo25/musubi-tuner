import assert from 'node:assert/strict';
import test from 'node:test';

import { estimateLatentCaching, estimateTextCaching, estimateTraining } from './vramEstimate.js';

function config(training = {}) {
	return {
		dataset: {
			datasets: [{ type: 'video', resolution_w: 832, resolution_h: 480, target_frames: 124, batch_size: 1 }],
		},
		training: {
			model_type: 'minimax_h3',
			ltx2_mode: 'av',
			network_dim: 16,
			gradient_checkpointing: true,
			...training,
		},
	};
}

function partValue(estimate, label) {
	return estimate.parts.find((part) => part.label === label)?.value;
}

test('H3 ConvRot INT8 reduces the estimated resident base weights', () => {
	const bf16 = estimateTraining(config());
	const fp8 = estimateTraining(config({ fp8_base: true }));
	const convrot = estimateTraining(config({ int8_convrot_base: true }));
	const onlineConvrot = estimateTraining(config({ h3_convrot_int8: true }));
	const onlineConvrotLowRank = estimateTraining(config({ h3_convrot_int8: true, h3_adaln_rank: 16 }));

	assert.equal(partValue(convrot, 'DiT'), 19.53);
	assert.ok(partValue(convrot, 'DiT') < partValue(fp8, 'DiT'));
	assert.ok(partValue(fp8, 'DiT') < partValue(bf16, 'DiT'));
	assert.ok(partValue(onlineConvrot, 'DiT') > partValue(fp8, 'DiT'));
	assert.ok(partValue(onlineConvrotLowRank, 'DiT') < partValue(fp8, 'DiT'));
});

test('H3 guidance distillation does not add a parallel activation peak', () => {
	const base = estimateTraining(config());
	const guided = estimateTraining(config({ h3_guidance_distillation_scale: 4.0 }));

	assert.equal(guided.total, base.total);
	assert.equal(partValue(guided, 'Guidance'), undefined);
});

test('H3 preservation changes average time but not estimated peak VRAM', () => {
	const base = estimateTraining(config({ h3_base_preservation_loss_weight: 0 }));
	const preserved = estimateTraining(config({ h3_base_preservation_loss_weight: 0.02 }));

	assert.equal(preserved.total, base.total);
	assert.equal(partValue(preserved, 'Preserv.'), undefined);
});

test('H3 partial checkpointing estimates more activation memory', () => {
	const full = estimateTraining(config());
	const partial = estimateTraining(config({ h3_gradient_checkpointing_blocks: 24 }));

	assert.ok(partValue(partial, 'Activ.') > partValue(full, 'Activ.'));
});

test('H3 text caching models Qwen3-VL quantization rather than Gemma', () => {
	const base = { caching: { model_type: 'minimax_h3', h3_text_encoder_quantization: 'none' } };
	const bf16 = estimateTextCaching(base);
	const nf4 = estimateTextCaching({ caching: { ...base.caching, h3_text_encoder_quantization: 'nf4' } });

	assert.equal(bf16.parts[0].label, 'Qwen3-VL 32B');
	assert.equal(partValue(bf16, 'Qwen3-VL 32B'), 47.97);
	assert.equal(partValue(nf4, 'Qwen3-VL 32B'), 14.0);
	assert.ok(bf16.total >= 51.97);
	assert.ok(nf4.total < bf16.total);
});

test('H3 long-sequence estimate stays near the measured 73.2 GiB workbox baseline', () => {
	const estimate = estimateTraining(config());

	// The production 100k-token BF16 checkpointed run peaked at 73.157 GiB.
	// Keep the UI conservative but within 10% of that measured baseline.
	assert.ok(estimate.total >= 73.157);
	assert.ok(estimate.total <= 73.157 * 1.10);
});

test('H3 estimators reject non-H3 configurations', () => {
	const ltx = { caching: { model_type: 'ltx2' }, training: { model_type: 'ltx2' } };
	assert.equal(estimateLatentCaching(ltx), null);
	assert.equal(estimateTextCaching(ltx), null);
	assert.equal(estimateTraining(ltx), null);
});
