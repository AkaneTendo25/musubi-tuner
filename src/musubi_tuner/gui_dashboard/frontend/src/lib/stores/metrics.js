import { writable } from 'svelte/store';
import { updateTick } from './sse.js';
import { loadParquet, query } from '../db.js';

export const lossData = writable([]);
export const lrData = writable([]);
export const stepTimeData = writable([]);
export const hasAudioLoss = writable(false);
export const dataLoading = writable(false);

let _unsub = null;

export function startMetricsPolling() {
	_unsub = updateTick.subscribe(async (tick) => {
		if (tick === 0) return; // skip initial
		await refreshMetrics();
	});

	// Initial load
	refreshMetrics();
}

export function stopMetricsPolling() {
	if (_unsub) {
		_unsub();
		_unsub = null;
	}
}

async function refreshMetrics() {
	dataLoading.set(true);
	try {
		const c = await loadParquet('/data/metrics.parquet');
		if (!c) {
			dataLoading.set(false);
			return;
		}

		const [loss, lr, st] = await Promise.all([
			query('SELECT step, loss, avr_loss, loss_v, loss_a FROM metrics ORDER BY step'),
			query('SELECT step, lr FROM metrics ORDER BY step'),
			query('SELECT step, step_time FROM metrics ORDER BY step')
		]);

		lossData.set(loss);
		lrData.set(lr);
		stepTimeData.set(st);

		// Check if audio loss data exists (non-NaN)
		const hasAudio = loss.some((r) => r.loss_a !== null && !isNaN(r.loss_a));
		hasAudioLoss.set(hasAudio);
	} catch (e) {
		console.warn('Failed to load metrics:', e);
	}
	dataLoading.set(false);
}
