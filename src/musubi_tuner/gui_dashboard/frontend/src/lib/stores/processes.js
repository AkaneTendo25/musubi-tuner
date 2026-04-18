import { writable } from 'svelte/store';
import { saveProjectNow } from '$lib/stores/project.js';

export const processStatuses = writable({
	cache_latents: { state: 'idle', exit_code: null },
	cache_text: { state: 'idle', exit_code: null },
	cache_dino: { state: 'idle', exit_code: null },
	training: { state: 'idle', exit_code: null },
	inference: { state: 'idle', exit_code: null },
	slider_training: { state: 'idle', exit_code: null }
});

export const processLogs = writable({
	cache_latents: [],
	cache_text: [],
	cache_dino: [],
	training: [],
	inference: [],
	slider_training: []
});

let _evtSource = null;
let _reconnectTimer = null;
let _pollTimer = null;

export function connectProcessSSE() {
	if (_evtSource) return;

	function open() {
		_evtSource = new EventSource('/sse/processes');

		_evtSource.addEventListener('status', (e) => {
			try {
				const data = JSON.parse(e.data);
				processStatuses.set(data);
			} catch { /* ignore */ }
		});

		_evtSource.addEventListener('logs', (e) => {
			try {
				const data = JSON.parse(e.data);
				processLogs.update((current) => {
					const updated = { ...current };
					if (!updated[data.type]) updated[data.type] = [];
					updated[data.type] = [...updated[data.type], ...data.lines];
					// Keep last 5000 lines
					if (updated[data.type].length > 5000) {
						updated[data.type] = updated[data.type].slice(-5000);
					}
					return updated;
				});
			} catch { /* ignore */ }
		});

		_evtSource.onerror = () => {
			_evtSource.close();
			_evtSource = null;
			_reconnectTimer = setTimeout(open, 3000);
		};
	}

	open();
}

export function disconnectProcessSSE() {
	if (_evtSource) {
		_evtSource.close();
		_evtSource = null;
	}
	if (_reconnectTimer) {
		clearTimeout(_reconnectTimer);
		_reconnectTimer = null;
	}
	if (_pollTimer) {
		clearInterval(_pollTimer);
		_pollTimer = null;
	}
}

export async function startProcess(type) {
	await saveProjectNow();

	const res = await fetch(`/api/processes/${type}/start`, { method: 'POST' });
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || `Failed to start ${type}`);
	}
	// Clear logs for fresh start
	processLogs.update((current) => ({ ...current, [type]: [] }));
	// Refresh status
	await refreshStatuses();
}

export async function stopProcess(type) {
	const res = await fetch(`/api/processes/${type}/stop`, { method: 'POST' });
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || `Failed to stop ${type}`);
	}
	await refreshStatuses();
}

export async function refreshStatuses() {
	try {
		const res = await fetch('/api/processes/status');
		if (res.ok) {
			const data = await res.json();
			processStatuses.set(data);
		}
	} catch { /* ignore */ }
}

export async function fetchLogs(type, lastN = null) {
	try {
		const url = lastN ? `/api/processes/${type}/logs?last_n=${lastN}` : `/api/processes/${type}/logs`;
		const res = await fetch(url);
		if (res.ok) {
			const data = await res.json();
			processLogs.update((current) => ({ ...current, [type]: data.lines }));
		}
	} catch { /* ignore */ }
}
