import { writable, derived } from 'svelte/store';
import { updateTick } from './sse.js';

export const status = writable(null);

let _unsub = null;
let _interval = null;
let _ignoreBeforeTime = 0;

export function clearStatus() {
	status.set(null);
}

export function ignoreStatusBefore(epochSeconds) {
	const numeric = Number(epochSeconds);
	_ignoreBeforeTime = Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
	clearStatus();
}

async function fetchStatus() {
	try {
		const res = await fetch('/data/status.json', { cache: 'no-store' });
		if (res.ok && res.status !== 204) {
			const payload = await res.json();
			const statusTime = Number(payload?.time || 0);
			if (_ignoreBeforeTime > 0 && (!Number.isFinite(statusTime) || statusTime < _ignoreBeforeTime)) {
				clearStatus();
				return;
			}
			status.set(payload);
		} else if (res.status === 204) {
			clearStatus();
		}
	} catch {
		// ignore
	}
}

export function startStatusPolling() {
	// Fetch status whenever SSE fires an update
	_unsub = updateTick.subscribe(async () => {
		await fetchStatus();
	});

	// Also poll periodically because status.json updates more often than metrics.parquet.
	_interval = setInterval(fetchStatus, 2000);

	// Also fetch immediately
	fetchStatus();
}

export function stopStatusPolling() {
	if (_unsub) {
		_unsub();
		_unsub = null;
	}
	if (_interval) {
		clearInterval(_interval);
		_interval = null;
	}
}

export const eta = derived(status, ($s) => {
	if (!$s || !$s.speed_steps_per_sec || $s.speed_steps_per_sec <= 0) return null;
	if (($s.step ?? 0) < 10) return null;
	if (($s.elapsed_sec ?? 0) < 45) return null;
	const remaining = ($s.max_steps || 0) - ($s.step || 0);
	if (remaining <= 0) return null;
	return remaining / $s.speed_steps_per_sec;
});
