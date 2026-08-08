import { joinPath } from './modelPaths.js';

export const CACHE_THEN_TRAIN_STAGES = [
	{
		id: 'dataset',
		label: 'Dataset',
		activeLabel: 'Preparing dataset',
		completeLabel: 'Dataset ready',
		processType: null,
		href: '/dataset',
		navLabel: 'Open Dataset'
	},
	{
		id: 'latents',
		label: 'Latents',
		activeLabel: 'Caching latents',
		completeLabel: 'Latents cached',
		processType: 'cache_latents',
		href: '/caching',
		navLabel: 'Open Caching'
	},
	{
		id: 'text',
		label: 'Text',
		activeLabel: 'Caching text',
		completeLabel: 'Text cached',
		processType: 'cache_text',
		href: '/caching',
		navLabel: 'Open Caching'
	},
	{
		id: 'training',
		label: 'Training',
		activeLabel: 'Training',
		completeLabel: 'Training complete',
		processType: 'training',
		href: '/training/dashboard',
		navLabel: 'Open Training'
	}
];

function cleanPath(value) {
	return typeof value === 'string' ? value.trim() : '';
}

function clampPercent(value) {
	const numeric = Number(value);
	if (!Number.isFinite(numeric)) return 0;
	return Math.max(0, Math.min(100, numeric));
}

function positiveNumber(value, fallback = 0) {
	const numeric = Number(value);
	return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
}

function formatInteger(value) {
	return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);
}

function plural(value, singular, pluralValue = `${singular}s`) {
	return value === 1 ? singular : pluralValue;
}

function formatHours(hours) {
	const numeric = Number(hours);
	if (!Number.isFinite(numeric) || numeric <= 0) return '';
	return `~${numeric.toFixed(numeric < 10 ? 1 : 0)} hrs`;
}

function formatCacheDuration(hours) {
	const numeric = Number(hours);
	if (!Number.isFinite(numeric) || numeric <= 0) return '';
	if (numeric < 1 / 60) return '<1 min';
	if (numeric < 1) return `~${Math.max(1, Math.round(numeric * 60))} min`;
	return formatHours(numeric);
}

function formatStepTime(seconds) {
	const numeric = Number(seconds);
	if (!Number.isFinite(numeric) || numeric <= 0) return '';
	if (numeric < 1) return `${Math.round(numeric * 1000)} ms/step`;
	return `${numeric.toFixed(numeric < 10 ? 1 : 0)} s/step`;
}

export function formatWorkflowDuration(hours) {
	const numeric = Number(hours);
	if (!Number.isFinite(numeric) || numeric <= 0) return '';
	if (numeric < 1) return `~${Math.max(1, Math.round(numeric * 60))} min`;
	if (numeric < 24) {
		const digits = numeric < 10 ? 1 : 0;
		return `~${numeric.toFixed(digits)} hrs`;
	}
	return `~${(numeric / 24).toFixed(1)} days`;
}

function normalizeStageWeights(units) {
	const raw = {
		latents: Math.max(Number(units?.latents) || 0, 0),
		text: Math.max(Number(units?.text) || 0, 0),
		training: Math.max(Number(units?.training) || 0, 0),
	};
	const total = raw.latents + raw.text + raw.training;
	if (total <= 0) {
		return { latents: 0.12, text: 0.08, training: 0.80 };
	}

	return {
		latents: raw.latents / total,
		text: raw.text / total,
		training: raw.training / total,
	};
}

function datasetMediaUnits(dataset) {
	const type = String(dataset?.type || 'video').toLowerCase();
	if (type === 'audio') return 0.9;
	if (type === 'image') return 0.55;

	const width = positiveNumber(dataset?.resolution_w, 768);
	const height = positiveNumber(dataset?.resolution_h, 512);
	const frames = positiveNumber(dataset?.target_frames, 33);
	const scale = (width * height * frames) / (768 * 512 * 33);
	return Math.max(0.8, Math.min(scale, 4.0));
}

export function estimateCacheThenTrainStageWeights(config = null) {
	if (!config) return normalizeStageWeights(null);

	const datasets = Array.isArray(config?.dataset?.datasets) ? config.dataset.datasets : [];
	const datasetCount = Math.max(datasets.length, 1);
	const mediaUnits = datasets.length > 0
		? datasets.reduce((sum, dataset) => sum + datasetMediaUnits(dataset), 0)
		: 1;
	const mode = String(config?.training?.ltx2_mode || config?.caching?.ltx2_mode || 'video').toLowerCase();
	const latentModeScale = mode === 'av' ? 1.25 : mode === 'audio' ? 1.1 : 1.0;
	const textModeScale = mode === 'audio' ? 0.8 : 1.0;
	const steps = positiveNumber(config?.training?.max_train_steps, 1600);

	return normalizeStageWeights({
		latents: Math.max(3.0, mediaUnits * 4.5 * latentModeScale),
		text: Math.max(1.4, datasetCount * 1.5 * textModeScale),
		training: Math.max(10.0, steps / 40),
	});
}

function estimateCacheHours(config = null, stats = null) {
	const datasets = Array.isArray(config?.dataset?.datasets) ? config.dataset.datasets : [];
	const items = positiveNumber(stats?.dataset?.total_items, 0);
	if (items <= 0) return 0;

	const mediaUnits = datasets.length > 0
		? datasets.reduce((sum, dataset) => sum + datasetMediaUnits(dataset), 0) / datasets.length
		: 1;
	const mode = String(config?.training?.ltx2_mode || config?.caching?.ltx2_mode || 'video').toLowerCase();
	const latentModeScale = mode === 'av' ? 1.35 : mode === 'audio' ? 1.1 : 1.0;
	const textModeScale = mode === 'audio' ? 0.75 : 1.0;
	const latentSeconds = items * mediaUnits * latentModeScale * 1.3;
	const textSeconds = (35 + items * 0.35) * textModeScale;
	return Math.max(0, (latentSeconds + textSeconds) / 3600);
}

export function buildCacheThenTrainIdleSummary(config = null, stats = null) {
	const trainingHours = positiveNumber(stats?.training?.estimated_time_hours, 0);
	const cacheHours = estimateCacheHours(config, stats);
	const totalDuration = formatHours(trainingHours + cacheHours);
	const trainingDuration = formatHours(trainingHours);
	const cacheDuration = formatCacheDuration(cacheHours);
	const steps = positiveNumber(config?.training?.max_train_steps, 0);
	const items = positiveNumber(stats?.dataset?.total_items, 0);
	const stepTime = formatStepTime(stats?.training?.estimated_step_time_sec);
	const datasets = Array.isArray(config?.dataset?.datasets) ? config.dataset.datasets.length : 0;
	const metrics = [];

	if (totalDuration) metrics.push({ label: 'Total', value: totalDuration });
	if (trainingDuration) metrics.push({ label: 'Training', value: trainingDuration });
	if (cacheDuration) metrics.push({ label: 'Cache', value: cacheDuration });
	if (steps > 0) metrics.push({ label: 'Steps', value: formatInteger(steps) });
	if (items > 0) metrics.push({ label: 'Items', value: formatInteger(items) });
	else if (datasets > 0) metrics.push({ label: 'Datasets', value: formatInteger(datasets) });
	if (stepTime) metrics.push({ label: 'Step Time', value: stepTime });

	const hasRuntime = trainingHours > 0 || cacheHours > 0;
	const hasDatasetScope = items > 0 || datasets > 0;

	return {
		label: hasRuntime ? 'Runtime Preview' : 'Workflow Setup',
		detail: hasRuntime
			? 'Based on current dataset and training settings'
			: hasDatasetScope
				? 'Add training duration data to complete the runtime preview'
				: 'Add a dataset to estimate cache and training runtime',
		metrics,
	};
}

export function isProcessActive(status) {
	return status?.state === 'running' || status?.state === 'stopping';
}

export function isProcessSuccessful(status) {
	return status?.state === 'finished' && (status.exit_code ?? 0) === 0;
}

export function defaultDatasetManifestPath(config) {
	return joinPath(config?.project_dir || '', 'dataset_manifest.json');
}

export function prepareCacheThenTrainConfig(config) {
	if (!config) return { config, manifestPath: '', changed: false };

	const caching = config.caching || {};
	const training = config.training || {};
	const cacheManifest = cleanPath(caching.save_dataset_manifest);
	const trainManifest = cleanPath(training.dataset_manifest);

	if (cacheManifest && trainManifest && cacheManifest !== trainManifest) {
		throw new Error('Caching and training are set to different dataset manifests.');
	}

	const manifestPath = cacheManifest || trainManifest || defaultDatasetManifestPath(config);
	const nextCaching = { ...caching, save_dataset_manifest: manifestPath };
	const nextTraining = { ...training, dataset_manifest: manifestPath };
	const changed =
		caching.save_dataset_manifest !== nextCaching.save_dataset_manifest ||
		training.dataset_manifest !== nextTraining.dataset_manifest;

	if (!changed) {
		return { config, manifestPath, changed: false };
	}

	return {
		config: {
			...config,
			caching: nextCaching,
			training: nextTraining,
		},
		manifestPath,
		changed: true,
	};
}

export function workflowStageFromProcessType(processType) {
	return CACHE_THEN_TRAIN_STAGES.find((stage) => stage.processType === processType) || null;
}

export function parseTqdmProgressLine(line) {
	if (typeof line !== 'string') return null;
	let match = line.trim().match(/(\d+)%\|.*?\|\s*(\d+)\/(\d+)/);
	if (!match) {
		match = line.trim().match(/^.*?(?::)?\s*(\d+)it\s+\[/i);
		if (!match) return null;
		const current = Number(match[1]);
		if (!Number.isFinite(current) || current < 0) return null;
		return {
			current,
			total: 0,
			percent: 0,
			label: `Iteration ${current}`
		};
	}

	const percent = clampPercent(Number(match[1]));
	const current = Number(match[2]);
	const total = Number(match[3]);
	if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return null;

	return {
		current,
		total,
		percent,
		label: `Batch ${current} of ${total}`
	};
}

export function latestTqdmProgress(logs) {
	if (!Array.isArray(logs)) return null;
	for (let i = logs.length - 1; i >= 0; i -= 1) {
		const parsed = parseTqdmProgressLine(logs[i]);
		if (parsed) return parsed;
	}
	return null;
}

export function cacheStatusProgress(progress) {
	if (!progress || typeof progress !== 'object') return null;

	const current = Number(progress.current_items || 0);
	const total = Number(progress.total_items || 0);
	if (!Number.isFinite(current) || current < 0) return null;

	if (Number.isFinite(total) && total > 0) {
		return {
			current,
			total,
			percent: clampPercent((current / total) * 100),
			label: `Item ${current} of ${total}`
		};
	}

	return {
		current,
		total: 0,
		percent: 0,
		label: current > 0 ? `${current} items cached` : 'Preparing items'
	};
}

export function statusLogProgress(progress) {
	if (!progress || typeof progress !== 'object') return null;

	const current = Number(progress.current || 0);
	const total = Number(progress.total || 0);
	if (!Number.isFinite(current) || current < 0) return null;

	const label = String(progress.label || '').trim().toLowerCase();
	const unit = String(progress.unit || '').trim().toLowerCase();
	if (Number.isFinite(total) && total > 0) {
		const prefix = unit === 'step' || label === 'steps' ? 'Step' : 'Batch';
		return {
			current,
			total,
			percent: clampPercent(progress.percent ?? ((current / total) * 100)),
			label: `${prefix} ${current} of ${total}`
		};
	}

	return {
		current,
		total: 0,
		percent: 0,
		label: `Iteration ${current}`
	};
}

export function activeWorkflowStageId(statuses = {}, fallbackStageId = '') {
	if (fallbackStageId && fallbackStageId !== 'idle' && fallbackStageId !== 'complete') {
		return fallbackStageId;
	}

	for (const stage of CACHE_THEN_TRAIN_STAGES) {
		if (isProcessActive(statuses?.[stage.processType])) return stage.id;
	}

	return '';
}

function stageProgress(stage, statuses, logs, trainingStatus, { ignoreSuccessful = false } = {}) {
	const status = statuses?.[stage.processType] || {};
	if (!ignoreSuccessful && isProcessSuccessful(status)) {
		return { percent: 100, detail: stage.completeLabel };
	}

	if (stage.processType === 'training') {
		const step = Number(trainingStatus?.step || 0);
		const maxSteps = Number(trainingStatus?.max_steps || 0);
		if (maxSteps > 0) {
			const detailParts = [`Step ${step} of ${maxSteps}`];
			if (trainingStatus?.loss != null) detailParts.push(`loss ${Number(trainingStatus.loss).toFixed(4)}`);
			return {
				percent: clampPercent((step / maxSteps) * 100),
				detail: detailParts.join(' - ')
			};
		}
		return { percent: 0, detail: 'Training started' };
	}

	const statusProgress = cacheStatusProgress(status.progress);
	const logStatusProgress = statusLogProgress(status.log_progress);
	const tqdmProgress = logStatusProgress || latestTqdmProgress(logs?.[stage.processType]);
	if (statusProgress && statusProgress.current <= 0 && statusProgress.total <= 0 && tqdmProgress) {
		return { ...tqdmProgress, detail: tqdmProgress.label };
	}
	const cacheProgress = statusProgress
		? {
			...statusProgress,
			percent: statusProgress.total > 0 ? statusProgress.percent : (tqdmProgress?.percent || statusProgress.percent)
		}
		: tqdmProgress;
	return cacheProgress ? { ...cacheProgress, detail: cacheProgress.label } : { percent: 0, detail: stage.activeLabel };
}

export function buildCacheThenTrainProgress({
	statuses = {},
	logs = {},
	trainingStatus = null,
	config = null,
	fallbackStageId = '',
	fallbackMessage = '',
	workflowRunning = false,
} = {}) {
	const activeId = activeWorkflowStageId(statuses, fallbackStageId);
	const activeIndex = CACHE_THEN_TRAIN_STAGES.findIndex((stage) => stage.id === activeId);
	const complete = fallbackStageId === 'complete';
	const stageWeights = estimateCacheThenTrainStageWeights(config);

	const stages = CACHE_THEN_TRAIN_STAGES.map((stage, index) => {
		let state = 'pending';
		const completedBeforeActive = activeIndex > index;
		if (complete || completedBeforeActive) {
			state = 'complete';
		} else if (stage.id === activeId) {
			state = !stage.processType || isProcessActive(statuses?.[stage.processType]) || workflowRunning ? 'active' : 'pending';
		}

		return { ...stage, state };
	});

	if (complete) {
		return {
			activeStage: null,
			stages,
			percent: 100,
			stagePercent: 100,
			label: 'Workflow complete',
			detail: 'Training started from the prepared cache.',
			navHref: '/training/dashboard',
			navLabel: 'Open Training',
		};
	}

	if (!activeId || activeIndex < 0) {
		return {
			activeStage: null,
			stages,
			percent: 0,
			stagePercent: 0,
			label: 'Ready',
			detail: fallbackMessage || 'Latents, text, then training.',
			navHref: '',
			navLabel: '',
		};
	}

	const activeStage = CACHE_THEN_TRAIN_STAGES[activeIndex];
	const activeStatus = statuses?.[activeStage.processType];
	const ignoreStaleSuccess = workflowRunning && activeId === fallbackStageId && !isProcessActive(activeStatus);
	const progress = stageProgress(activeStage, statuses, logs, trainingStatus, { ignoreSuccessful: ignoreStaleSuccess });
	const stagePercent = clampPercent(progress.percent);
	const completedWeight = CACHE_THEN_TRAIN_STAGES
		.slice(0, activeIndex)
		.reduce((sum, stage) => sum + (stageWeights[stage.id] || 0), 0);
	const percent = clampPercent((completedWeight + (stageWeights[activeStage.id] || 0) * (stagePercent / 100)) * 100);

	return {
		activeStage,
		stages,
		percent,
		stagePercent,
		label: fallbackMessage || activeStage.activeLabel,
		detail: progress.detail || activeStage.activeLabel,
		navHref: activeStage.href,
		navLabel: activeStage.navLabel,
	};
}
