function formatDownloadError(detail) {
	if (!detail) return 'Download failed';
	if (typeof detail === 'string') return detail;
	if (Array.isArray(detail)) {
		return detail.map((item) => item?.msg || item?.detail || JSON.stringify(item)).join('; ');
	}
	if (typeof detail === 'object') {
		return detail.detail || detail.msg || JSON.stringify(detail);
	}
	return String(detail);
}

async function parseJsonOrEmpty(res) {
	return res.json().catch(() => ({}));
}

function formatBytes(bytes) {
	if (!Number.isFinite(bytes) || bytes <= 0) return '';
	const units = ['B', 'KB', 'MB', 'GB', 'TB'];
	let value = bytes;
	let index = 0;
	while (value >= 1024 && index < units.length - 1) {
		value /= 1024;
		index += 1;
	}
	return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

export async function startModelDownload(preset, targetPath) {
	const res = await fetch('/api/fs/download-model', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ preset, dest_path: targetPath })
	});
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return data;
}

export async function getModelDownloadPreflight(preset, targetPath) {
	const res = await fetch('/api/fs/download-preflight', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ preset, dest_path: targetPath })
	});
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return data;
}

export async function getModelDownloadPresets() {
	const res = await fetch('/api/fs/download-presets');
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return data.presets || {};
}

export async function scanCheckpoints(type, extraPaths = '') {
	const params = new URLSearchParams({ type, extra_paths: extraPaths || '' });
	const res = await fetch(`/api/fs/scan-checkpoints?${params}`);
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return data.results || [];
}

export async function checkPathExists(path) {
	if (!path) return false;
	const params = new URLSearchParams({ path });
	const res = await fetch(`/api/fs/exists?${params}`);
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return Boolean(data.exists);
}

export function modelDownloadUrl(downloadPresets, preset) {
	return downloadPresets?.[preset]?.url || '';
}

export function modelDownloadTooltip(downloadPresets, preset, targetPath, targetExists = false) {
	const url = modelDownloadUrl(downloadPresets, preset);
	if (targetExists) {
		return `Already exists: ${targetPath}`;
	}
	return url ? `Download ${url} to ${targetPath}` : `Download to ${targetPath}`;
}

export function formatModelPreflightStatus(preflight) {
	if (!preflight) return '';
	if (preflight.errors?.length) return preflight.errors.join('; ');
	const total = formatBytes(preflight.total_bytes) || 'unknown size';
	const free = formatBytes(preflight.free_bytes) || 'unknown free space';
	const warning = preflight.warnings?.length ? ` (${preflight.warnings.join('; ')})` : '';
	return `Preflight OK: ${total}, ${free} available${warning}`;
}

export async function getModelDownloadStatus(jobId) {
	const res = await fetch(`/api/fs/download-model/${jobId}`);
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return data;
}

export async function cancelModelDownload(jobId) {
	const res = await fetch(`/api/fs/download-model/${jobId}/cancel`, { method: 'POST' });
	const data = await parseJsonOrEmpty(res);
	if (!res.ok) {
		throw new Error(formatDownloadError(data.detail || data));
	}
	return data;
}

export function isActiveModelDownload(status) {
	return ['queued', 'running', 'cancelling'].includes(status?.state || '');
}

export function formatModelDownloadStatus(status) {
	if (!status) return '';
	const downloaded = formatBytes(status.bytes_downloaded);
	const total = formatBytes(status.total_bytes);
	const progress = downloaded && total ? ` ${downloaded} / ${total}` : downloaded ? ` ${downloaded}` : '';

	switch (status.state) {
		case 'queued':
			return status.message || 'Queued';
		case 'running':
			return `${status.message || 'Downloading'}${progress}`;
		case 'cancelling':
			return status.message || 'Cancelling download';
		case 'completed':
			return status.message || (status.path ? `Saved to ${status.path}` : 'Download complete');
		case 'cancelled':
			return status.message || 'Download cancelled';
		case 'failed':
			return status.error || status.message || 'Download failed';
		default:
			return status.message || '';
	}
}

export function getModelDownloadTone(status) {
	switch (status?.state) {
		case 'completed':
			return 'success';
		case 'failed':
			return 'danger';
		case 'cancelled':
			return 'muted';
		case 'queued':
		case 'running':
		case 'cancelling':
			return 'accent';
		default:
			return 'muted';
	}
}
