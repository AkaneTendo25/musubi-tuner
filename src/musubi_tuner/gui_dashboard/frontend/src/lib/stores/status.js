import { writable, derived } from 'svelte/store';
import { updateTick } from './sse.js';

export const status = writable(null);

let _unsub = null;
let _interval = null;

export function clearStatus() {
	status.set(null);
}

async function fetchStatus() {
	try {
		const res = await fetch('/data/status.json', { cache: 'no-store' });
		if (res.ok && res.status !== 204) {
			status.set(await res.json());
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
