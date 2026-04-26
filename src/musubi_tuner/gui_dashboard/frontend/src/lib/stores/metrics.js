import { writable } from 'svelte/store';
import { updateTick } from './sse.js';

export const lossData = writable([]);
export const lrData = writable([]);
export const stepTimeData = writable([]);
export const gradNormData = writable([]);
export const dataWaitData = writable([]);
export const validationLossData = writable([]);
export const hasAudioLoss = writable(false);
export const dataLoading = writable(false);

let _unsub = null;
let _interval = null;
let _lastMetricsSignature = null;
let _lastEventsSignature = null;
let _hasLoadedOnce = false;

export function clearMetrics() {
	lossData.set([]);
	lrData.set([]);
	stepTimeData.set([]);
	gradNormData.set([]);
	dataWaitData.set([]);
	validationLossData.set([]);
	hasAudioLoss.set(false);
	dataLoading.set(false);
	_lastMetricsSignature = null;
	_lastEventsSignature = null;
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
				gradNormData.set([]);
				dataWaitData.set([]);
				validationLossData.set([]);
				hasAudioLoss.set(false);
				_lastMetricsSignature = null;
				_lastEventsSignature = null;
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
				gradNormData.set([]);
				dataWaitData.set([]);
				validationLossData.set([]);
				hasAudioLoss.set(false);
				_lastMetricsSignature = null;
				_lastEventsSignature = null;
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
			last?.grad_norm ?? '',
			last?.grad_norm_v ?? '',
			last?.grad_norm_a ?? '',
			last?.lr ?? '',
			last?.step_time ?? '',
			last?.data_wait_time ?? ''
		].join('|');
		if (signature !== _lastMetricsSignature) {
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
			const grad = rows.map((r) => ({
				step: r.step,
				grad_norm: r.grad_norm,
				grad_norm_v: r.grad_norm_v,
				grad_norm_a: r.grad_norm_a
			}));
			const wait = rows.map((r) => ({
				step: r.step,
				data_wait_time: r.data_wait_time
			}));

			lossData.set(loss);
			lrData.set(lr);
			stepTimeData.set(st);
			gradNormData.set(grad);
			dataWaitData.set(wait);

			// Check if audio loss data exists (non-NaN)
			const hasAudio = loss.some((r) => r.loss_a !== null && !isNaN(r.loss_a));
			hasAudioLoss.set(hasAudio);
		}

		await refreshValidationEvents();
	} catch (e) {
		console.warn('Failed to load metrics:', e);
	}
	_hasLoadedOnce = true;
	dataLoading.set(false);
}

async function refreshValidationEvents() {
	try {
		const eventsRes = await fetch('/data/events.json', { cache: 'no-store' });
		if (eventsRes.ok && eventsRes.status !== 204) {
			const events = await eventsRes.json();
			const valEvents = events.filter((e) => e.type === 'validation' && typeof e.val_loss === 'number');
			const lastVal = valEvents[valEvents.length - 1];
			const eventsSignature = [valEvents.length, lastVal?.step ?? '', lastVal?.val_loss ?? ''].join('|');
			if (eventsSignature !== _lastEventsSignature) {
				_lastEventsSignature = eventsSignature;
				validationLossData.set(
					valEvents.map((e) => ({
						step: e.step,
						val_loss: e.val_loss,
						epoch: e.epoch ?? null
					}))
				);
			}
		} else if (_lastEventsSignature !== null) {
			validationLossData.set([]);
			_lastEventsSignature = null;
		}
	} catch (e) {
		console.warn('Failed to load dashboard events:', e);
	}
}
