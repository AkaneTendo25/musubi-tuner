import { get, writable } from 'svelte/store';
import { projectConfig, saveProjectNow } from './project.js';
import {
	cancelModelDownload,
	formatModelDownloadStatus,
	formatModelPreflightStatus,
	getModelDownloadPreflight,
	getModelDownloadStatus,
	getModelDownloadTone,
	isActiveModelDownload,
	startModelDownload
} from '$lib/utils/modelDownloads.js';

const STORAGE_KEY = 'musubi.modelDownload.state';
const ACTIVE_STATES = new Set(['queued', 'running', 'cancelling']);

function initialState() {
	if (typeof sessionStorage === 'undefined') {
		return emptyState();
	}
	try {
		return { ...emptyState(), ...(JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '{}') || {}) };
	} catch {
		return emptyState();
	}
}

function emptyState() {
	return {
		jobId: '',
		preset: '',
		section: '',
		targetPath: '',
		modelDir: '',
		state: '',
		message: '',
		tone: 'muted',
		error: '',
		path: ''
	};
}

export const modelDownloadState = writable(initialState());

modelDownloadState.subscribe((state) => {
	if (typeof sessionStorage === 'undefined') return;
	try {
		sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state || emptyState()));
	} catch {}
});

let pollTimer = null;

function setStatus(downloadStatus, patch = {}) {
	modelDownloadState.update((current) => ({
		...current,
		...patch,
		state: downloadStatus?.state || '',
		message: formatModelDownloadStatus(downloadStatus),
		tone: getModelDownloadTone(downloadStatus),
		error: downloadStatus?.error || '',
		path: downloadStatus?.path || current.path || '',
		jobId: downloadStatus?.job_id || patch.jobId || current.jobId || ''
	}));
}

function clearActiveJob() {
	modelDownloadState.update((current) => ({
		...current,
		jobId: '',
		state: current.state || '',
	}));
}

async function applyCompletedDownload(downloadStatus, context) {
	if (downloadStatus?.state !== 'completed' || !downloadStatus.path) return;
	projectConfig.update((config) => {
		if (!config) return config;
		const section = context.section || 'training';
		const currentSection = { ...(config[section] || {}) };
		const next = { ...config };
		if (context.modelDir) next.model_dir = context.modelDir;
		if (context.preset === 'ltxav') {
			next.default_ltx2_checkpoint = downloadStatus.path;
			currentSection.ltx2_checkpoint = downloadStatus.path;
		} else if (context.preset === 'gemma-unsloth') {
			next.default_gemma_root = downloadStatus.path;
			next.default_gemma_safetensors = '';
			currentSection.gemma_root = downloadStatus.path;
			currentSection.gemma_safetensors = '';
		}
		next[section] = currentSection;
		return next;
	});
	await saveProjectNow();
}

export function hasActiveModelDownload(state = get(modelDownloadState)) {
	return Boolean(state?.jobId) && ACTIVE_STATES.has(state?.state || '');
}

export async function resumeModelDownloadPolling() {
	const state = get(modelDownloadState);
	if (!hasActiveModelDownload(state) || pollTimer) return;
	await pollSharedModelDownload(state.jobId);
}

export async function pollSharedModelDownload(jobId = get(modelDownloadState).jobId) {
	if (pollTimer) clearTimeout(pollTimer);
	if (!jobId) return;
	try {
		const downloadStatus = await getModelDownloadStatus(jobId);
		const context = get(modelDownloadState);
		setStatus(downloadStatus);
		if (!isActiveModelDownload(downloadStatus)) {
			await applyCompletedDownload(downloadStatus, context);
			clearActiveJob();
			return;
		}
		pollTimer = setTimeout(() => {
			pollTimer = null;
			void pollSharedModelDownload(jobId);
		}, 1000);
	} catch (e) {
		setStatus({ state: 'failed', error: e?.message || 'Download status failed' });
		clearActiveJob();
	}
}

export async function startSharedModelDownload({ preset, targetPath, modelDir = '', section = 'training' }) {
	if (hasActiveModelDownload()) return;
	if (!preset || !targetPath) return;
	modelDownloadState.update((current) => ({
		...current,
		preset,
		section,
		targetPath,
		modelDir,
		path: '',
		error: ''
	}));
	try {
		const preflight = await getModelDownloadPreflight(preset, targetPath);
		setStatus({
			state: preflight.ok ? 'queued' : 'failed',
			message: formatModelPreflightStatus(preflight),
			error: preflight.errors?.join('; ')
		});
		if (!preflight.ok) return;
		const job = await startModelDownload(preset, targetPath);
		setStatus(job, {
			jobId: job.job_id || '',
			preset,
			section,
			targetPath,
			modelDir
		});
		if (job.job_id) await pollSharedModelDownload(job.job_id);
	} catch (e) {
		setStatus({ state: 'failed', error: e?.message || 'Download failed' });
		clearActiveJob();
	}
}

export async function cancelSharedModelDownload() {
	const state = get(modelDownloadState);
	if (!state.jobId) return;
	try {
		const downloadStatus = await cancelModelDownload(state.jobId);
		setStatus(downloadStatus);
		if (!isActiveModelDownload(downloadStatus)) clearActiveJob();
	} catch (e) {
		setStatus({ state: 'failed', error: e?.message || 'Cancel failed' });
		clearActiveJob();
	}
}
