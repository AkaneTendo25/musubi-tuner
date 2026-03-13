import { writable, derived } from 'svelte/store';
import { updateTick } from './sse.js';

export const status = writable(null);

let _unsub = null;

export function startStatusPolling() {
	// Fetch status whenever SSE fires an update
	_unsub = updateTick.subscribe(async () => {
		try {
			const res = await fetch('/data/status.json');
			if (res.ok && res.status !== 204) {
				status.set(await res.json());
			}
		} catch {
			// ignore
		}
	});

	// Also fetch immediately
	fetch('/data/status.json')
		.then((r) => (r.ok && r.status !== 204 ? r.json() : null))
		.then((d) => { if (d) status.set(d); })
		.catch(() => {});
}

export function stopStatusPolling() {
	if (_unsub) {
		_unsub();
		_unsub = null;
	}
}

export const eta = derived(status, ($s) => {
	if (!$s || !$s.speed_steps_per_sec || $s.speed_steps_per_sec <= 0) return null;
	const remaining = ($s.max_steps || 0) - ($s.step || 0);
	if (remaining <= 0) return null;
	return remaining / $s.speed_steps_per_sec;
});
