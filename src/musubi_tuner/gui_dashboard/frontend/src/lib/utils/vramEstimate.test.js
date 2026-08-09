import assert from 'node:assert/strict';
import test from 'node:test';

import { estimateTraining } from './vramEstimate.js';

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

	assert.equal(partValue(convrot, 'DiT'), 19.53);
	assert.ok(partValue(convrot, 'DiT') < partValue(fp8, 'DiT'));
	assert.ok(partValue(fp8, 'DiT') < partValue(bf16, 'DiT'));
});

test('H3 partial checkpointing estimates more activation memory', () => {
	const full = estimateTraining(config());
	const partial = estimateTraining(config({ h3_gradient_checkpointing_blocks: 24 }));

	assert.ok(partValue(partial, 'Activ.') > partValue(full, 'Activ.'));
});

test('LTX-2 base estimates retain their existing constants', () => {
	const bf16 = estimateTraining(config({ model_type: 'ltx2', ltx_version: '2.3', ltx2_mode: 'video' }));
	const fp8 = estimateTraining(config({ model_type: 'ltx2', ltx_version: '2.3', ltx2_mode: 'video', fp8_base: true }));

	assert.equal(partValue(bf16, 'DiT'), 42);
	assert.equal(partValue(fp8, 'DiT'), 21);
});
