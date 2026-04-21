import { writable } from 'svelte/store';
import { get } from 'svelte/store';
import { projectConfig, saveProjectNow } from '$lib/stores/project.js';
import { clearMetrics } from '$lib/stores/metrics.js';
import { clearStatus } from '$lib/stores/status.js';

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

export const processConsoleUi = writable({
	cache_latents: { collapsed: null },
	cache_text: { collapsed: null },
	cache_dino: { collapsed: null },
	training: { collapsed: null },
	inference: { collapsed: null },
	slider_training: { collapsed: null }
});

function emptyValidation() {
	return {
		ok: true,
		summary: '',
		errors: [],
		warnings: [],
		field_errors: {},
		field_warnings: {}
	};
}

export const processValidation = writable({
	cache_latents: emptyValidation(),
	cache_text: emptyValidation(),
	cache_dino: emptyValidation(),
	training: emptyValidation(),
	inference: emptyValidation(),
	slider_training: emptyValidation()
});

let _evtSource = null;
let _reconnectTimer = null;

function isActiveState(state) {
	return state === 'running' || state === 'stopping';
}

function normalizeValidationReport(payload) {
	if (!payload || typeof payload !== 'object') {
		return null;
	}

	return {
		ok: payload.ok ?? false,
		summary: payload.summary || '',
		errors: Array.isArray(payload.errors) ? payload.errors : [],
		warnings: Array.isArray(payload.warnings) ? payload.warnings : [],
		field_errors: payload.field_errors || {},
		field_warnings: payload.field_warnings || {}
	};
}

function extractErrorMessage(payload, fallback) {
	if (!payload) return fallback;
	if (typeof payload.detail === 'string') return payload.detail;
	if (typeof payload.detail?.summary === 'string') return payload.detail.summary;
	if (typeof payload.summary === 'string') return payload.summary;
	return fallback;
}

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
}

export function clearProcessLogs(type = null) {
	processLogs.update((current) => {
		if (!type) {
			return {
				cache_latents: [],
				cache_text: [],
				cache_dino: [],
				training: [],
				inference: [],
				slider_training: []
			};
		}
		return { ...current, [type]: [] };
	});
}

export function setProcessConsoleCollapsed(type, collapsed) {
	processConsoleUi.update((current) => ({
		...current,
		[type]: {
			...(current[type] || {}),
			collapsed
		}
	}));
}

function setProcessStatus(type, status) {
	processStatuses.update((current) => ({
		...current,
		[type]: status
	}));
}

export async function validateProcess(type, configOverride = null) {
	const init = { method: 'POST' };
	if (configOverride) {
		init.headers = { 'Content-Type': 'application/json' };
		init.body = JSON.stringify(configOverride);
	}

	const res = await fetch(`/api/processes/${type}/validate`, init);
	const payload = await res.json().catch(() => null);
	const report = normalizeValidationReport(res.ok ? payload : payload?.detail || payload);

	if (report) {
		processValidation.update((current) => ({ ...current, [type]: report }));
	}

	if (!res.ok) {
		throw new Error(extractErrorMessage(payload, `Failed to validate ${type}`));
	}

	return report || emptyValidation();
}

export async function startProcess(type) {
	await saveProjectNow();
	await validateProcess(type, get(projectConfig));
	if (type === 'training') {
		clearMetrics();
		clearStatus();
		clearProcessLogs(type);
		setProcessStatus(type, { state: 'running', exit_code: null });
	}

	const res = await fetch(`/api/processes/${type}/start`, { method: 'POST' });
	if (!res.ok) {
		await refreshStatuses();
		const err = await res.json().catch(() => null);
		const report = normalizeValidationReport(err?.detail || err);
		if (report) {
			processValidation.update((current) => ({ ...current, [type]: report }));
		}
		throw new Error(extractErrorMessage(err, `Failed to start ${type}`));
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
			return data;
		}
	} catch { /* ignore */ }
	return null;
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

export async function preloadLogsIfActive(types) {
	const targets = Array.isArray(types) ? types : [types];
	const statuses = (await refreshStatuses()) || get(processStatuses);

	for (const type of targets) {
		const state = statuses?.[type]?.state;
		if (isActiveState(state)) {
			await fetchLogs(type);
		} else {
			clearProcessLogs(type);
		}
	}
}

export function startLogPolling(types, intervalMs = 1000) {
	const targets = Array.isArray(types) ? types : [types];

	const poll = async () => {
		const statuses = get(processStatuses);
		for (const type of targets) {
			const state = statuses?.[type]?.state;
			if (isActiveState(state)) {
				await fetchLogs(type);
			}
		}
	};

	poll();
	return setInterval(poll, intervalMs);
}
