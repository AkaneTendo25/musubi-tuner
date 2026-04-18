import { writable } from 'svelte/store';
import { updateTick } from './sse.js';

export const lossData = writable([]);
export const lrData = writable([]);
export const stepTimeData = writable([]);
export const hasAudioLoss = writable(false);
export const dataLoading = writable(false);

let _unsub = null;
let _interval = null;
let _lastMetricsSignature = null;
let _hasLoadedOnce = false;

export function clearMetrics() {
	lossData.set([]);
	lrData.set([]);
	stepTimeData.set([]);
	hasAudioLoss.set(false);
	dataLoading.set(false);
	_lastMetricsSignature = null;
	_hasLoadedOnce = false;
}

export function startMetricsPolling() {
	_unsub = updateTick.subscribe(async (tick) => {
		if (tick === 0) return; // skip initial
		await refreshMetrics();
	});

	_interval = setInterval(refreshMetrics, 2000);

	// Initial load
	refreshMetrics();
}

export function stopMetricsPolling() {
	if (_unsub) {
		_unsub();
		_unsub = null;
	}
	if (_interval) {
		clearInterval(_interval);
		_interval = null;
	}
}

async function refreshMetrics() {
	if (!_hasLoadedOnce) dataLoading.set(true);
	try {
		const res = await fetch('/api/dashboard/metrics', { cache: 'no-store' });
		if (!res.ok || res.status === 204) {
			if (_lastMetricsSignature !== null) {
				lossData.set([]);
				lrData.set([]);
				stepTimeData.set([]);
				hasAudioLoss.set(false);
				_lastMetricsSignature = null;
			}
			_hasLoadedOnce = true;
			dataLoading.set(false);
			return;
		}

		const payload = await res.json();
		const rows = Array.isArray(payload?.rows) ? payload.rows : [];
		if (!rows.length) {
			if (_lastMetricsSignature !== null) {
				lossData.set([]);
				lrData.set([]);
				stepTimeData.set([]);
				hasAudioLoss.set(false);
				_lastMetricsSignature = null;
			}
			_hasLoadedOnce = true;
			dataLoading.set(false);
			return;
		}

		const last = rows[rows.length - 1];
		const signature = [
			rows.length,
			last?.step ?? '',
			last?.loss ?? '',
			last?.avr_loss ?? '',
			last?.loss_v ?? '',
			last?.loss_a ?? '',
			last?.lr ?? '',
			last?.step_time ?? ''
		].join('|');
		if (signature === _lastMetricsSignature) {
			_hasLoadedOnce = true;
			dataLoading.set(false);
			return;
		}
		_lastMetricsSignature = signature;

		const loss = rows.map((r) => ({
			step: r.step,
			loss: r.loss,
			avr_loss: r.avr_loss,
			loss_v: r.loss_v,
			loss_a: r.loss_a
		}));
		const lr = rows.map((r) => ({ step: r.step, lr: r.lr }));
		const st = rows.map((r) => ({ step: r.step, step_time: r.step_time }));

		lossData.set(loss);
		lrData.set(lr);
		stepTimeData.set(st);

		// Check if audio loss data exists (non-NaN)
		const hasAudio = loss.some((r) => r.loss_a !== null && !isNaN(r.loss_a));
		hasAudioLoss.set(hasAudio);
	} catch (e) {
		console.warn('Failed to load metrics:', e);
	}
	_hasLoadedOnce = true;
	dataLoading.set(false);
}
