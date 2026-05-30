<script>
	import FormField from '$lib/components/FormField.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import StatsPanel from '$lib/components/StatsPanel.svelte';
	import VramEstimateCard from '$lib/components/VramEstimateCard.svelte';
	import { CACHE_THEN_TRAIN_STAGES, buildCacheThenTrainIdleSummary, buildCacheThenTrainProgress, isProcessActive, prepareCacheThenTrainConfig } from '$lib/utils/cacheWorkflow.js';
	import { defaultModelDir, effectiveGemmaRoot, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { estimateLatentCaching as estimateLatentCachingVram, estimateTextCaching as estimateTextCachingVram, estimateTraining as estimateTrainingVram } from '$lib/utils/vramEstimate.js';
	import { projectConfig, projectLoaded, createProject, loadProjectFromPath, saveProjectDebounced, saveProjectNow, recentProjects, removeRecentProject, restoreRecentProject, closeProject } from '$lib/stores/project.js';
	import { processLogs, processStatuses, preloadLogsIfActive, startLogPolling, startProcess, stopProcess, refreshStatuses } from '$lib/stores/processes.js';
	import { status } from '$lib/stores/status.js';
	import { onMount } from 'svelte';

	let newProjectName = $state('New Project');
	let loadPath = $state('');
	let error = $state('');
	let creating = $state(false);
	let projectDirExists = $state(false);
	let projectDirCheckPending = $state(false);
	let systemInfo = $state(null);
	let cwd = $state('');
	let selectedLoraFamily = $state('video');
	let selectedMemoryProfile = $state('regular');
	let removedRecentProjects = $state([]);
	let recentProjectSummaries = $state({});
	let recentProjectSummaryLoading = $state({});
	let cacheThenTrainRunning = $state(false);
	let cacheThenTrainMessage = $state('');
	let cacheThenTrainError = $state('');
	let cacheThenTrainPhase = $state('idle');
	let cacheThenTrainEngaged = $state(false);
	let cacheThenTrainStopRequested = $state(false);
	let cacheThenTrainStopPending = $state(false);
	let workflowStats = $state(null);
	let workflowStatsLoading = $state(false);
	let workflowStatsSnapshot = '';
	let workflowStatsTimer = null;

	const LTX_DOCS_FALLBACK_URL = 'https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2/docs/ltx_2.md';
	const TEMPLATE_VARIANTS_DISABLED = true;
	const CACHE_THEN_TRAIN_SESSION_KEY = 'musubi.cacheThenTrain.engaged';

	const LORA_FAMILIES = [
		{
			id: 'video',
			label: 'Video',
			blurb: 'Standard visual LoRA setup for image and video datasets.'
		},
		{
			id: 'av',
			label: 'Audio-Video',
			blurb: 'Joint audio and video training defaults for synchronized clips.'
		},
		{
			id: 'audio',
			label: 'Audio-Only',
			blurb: 'Audio-only mode with audio-target defaults already selected.'
		},
		{
			id: 'ic',
			label: 'IC-LoRA',
			blurb: 'Reference-conditioned setup for identity or shared-reference work.'
		},
		{
			id: 'slider',
			label: 'Slider',
			blurb: 'Base project for prompt or reference sliders from the Techniques page.'
		}
	];

	const MEMORY_PROFILES = [
		{
			id: 'high',
			label: 'High VRAM',
			blurb: 'Full-weight defaults with minimal memory-saving tradeoffs.'
		},
		{
			id: 'regular',
			label: 'Regular',
			blurb: 'Moderate memory savings with 8-bit Gemma and checkpointing enabled.'
		},
		{
			id: 'low',
			label: 'Low VRAM',
			blurb: 'FP8 / 4-bit oriented defaults for tighter cards.'
		}
	];

	let activeLoraFamily = $derived(LORA_FAMILIES.find((item) => item.id === selectedLoraFamily) || LORA_FAMILIES[0]);
	let activeMemoryProfile = $derived(MEMORY_PROFILES.find((item) => item.id === selectedMemoryProfile) || MEMORY_PROFILES[1]);
	let ltxDocsUrl = $derived(systemInfo?.repo?.docs_url || LTX_DOCS_FALLBACK_URL);
	let ltxDocsBranch = $derived(systemInfo?.repo?.docs_branch || systemInfo?.repo?.branch || 'ltx-2');
	let repoRoot = $derived(systemInfo?.repo?.root || cwd);
	let newProjectSlug = $derived(slugProjectName(newProjectName));
	let newProjectDir = $derived(repoRoot && newProjectSlug ? `${repoRoot}/projects/${newProjectSlug}` : '');

	let projectDirCheckSeq = 0;

	onMount(async () => {
		try {
			const [infoRes, cwdRes] = await Promise.all([
				fetch('/api/system/info'),
				fetch('/api/fs/cwd'),
			]);
			if (infoRes.ok) systemInfo = await infoRes.json();
			if (cwdRes.ok) {
				cwd = (await cwdRes.json()).cwd || '';
			}
		} catch {}
	});

	onMount(() => {
		const processTypes = CACHE_THEN_TRAIN_STAGES.map((stage) => stage.processType).filter(Boolean);
		let logTimer = null;
		const statusTimer = setInterval(() => {
			void refreshStatuses();
		}, 1000);
		void (async () => {
			const statuses = await refreshStatuses();
			if (readCacheThenTrainSession() && hasActiveWorkflowProcess(statuses || $processStatuses)) {
				cacheThenTrainEngaged = true;
			}
			await preloadLogsIfActive(processTypes);
			logTimer = startLogPolling(processTypes, 1000);
		})();
		return () => {
			clearInterval(statusTimer);
			if (logTimer) clearInterval(logTimer);
		};
	});

	$effect(() => {
		if ($projectLoaded) return;
		for (const project of $recentProjects) {
			const path = project?.path;
			if (!path || recentProjectSummaries[path] || recentProjectSummaryLoading[path]) continue;
			void loadRecentProjectSummary(path);
		}
	});

	$effect(() => {
		if (!cacheThenTrainEngaged || cacheThenTrainRunning) return;
		if (!hasActiveWorkflowProcess($processStatuses)) {
			setCacheThenTrainEngaged(false);
		}
	});

	$effect(() => {
		const path = newProjectDir;
		if (!path) {
			projectDirExists = false;
			projectDirCheckPending = false;
			return;
		}
		const seq = ++projectDirCheckSeq;
		projectDirCheckPending = true;
		fetch(`/api/fs/exists?path=${encodeURIComponent(path)}`)
			.then((res) => res.ok ? res.json() : { exists: false })
			.then((data) => {
				if (seq === projectDirCheckSeq) projectDirExists = !!data.exists;
			})
			.catch(() => {
				if (seq === projectDirCheckSeq) projectDirExists = false;
			})
			.finally(() => {
				if (seq === projectDirCheckSeq) projectDirCheckPending = false;
			});
	});

	function slugProjectName(name) {
		const slug = (name || '')
			.trim()
			.toLowerCase()
			.replace(/[^\w\s-]/g, '')
			.replace(/[\s-]+/g, '_');
		return slug || 'project';
	}

	function buildTemplateConfig(loraFamily, memoryProfile, defaultLtx, defaultGemma) {
		const repoOutputDir = repoRoot ? `${repoRoot}/output/${newProjectSlug}` : `output/${newProjectSlug}`;
		const repoLoggingDir = repoRoot ? `${repoRoot}/logs/${newProjectSlug}` : `logs/${newProjectSlug}`;
		const shared = {
			version: 2,
			default_ltx2_checkpoint: defaultLtx,
			default_gemma_root: defaultGemma,
			default_gemma_safetensors: '',
			caching: {
				ltx2_checkpoint: defaultLtx,
				gemma_root: defaultGemma,
				ltx2_mode: 'video',
				mixed_precision: 'bf16',
			},
			training: {
				ltx2_checkpoint: defaultLtx,
				gemma_root: defaultGemma,
				ltx2_mode: 'video',
				ltx_version: '2.3',
				mixed_precision: 'bf16',
				fp8_base: true,
				fp8_scaled: true,
				sdpa: true,
				sample_sampling_preset: 'defaults',
				lora_target_preset: 't2v',
				ic_lora_strategy: 'auto',
				network_dim: 32,
				network_alpha: 32,
				optimizer_type: 'adamw8bit',
				learning_rate: 1e-4,
				gradient_checkpointing: true,
				timestep_sampling: 'shifted_logit_normal',
				output_dir: repoOutputDir,
				autoresume: true,
				save_every_n_steps: 400,
				save_state: true,
				logging_dir: repoLoggingDir,
				accelerate_extra_args: '--num_processes 1 --num_machines 1 --num_cpu_threads_per_process 1',
				max_data_loader_n_workers: 1,
			},
			inference: {
				ltx2_checkpoint: defaultLtx,
				gemma_root: defaultGemma,
				ltx2_mode: 'video',
				sampling_preset: 'defaults',
			},
			slider: {
				mode: 'text',
				output_name: 'ltx2_slider',
				accelerate_extra_args: '--num_processes 1 --num_machines 1 --num_cpu_threads_per_process 1',
			},
		};

		let familyConfig = shared;
		if (loraFamily === 'av') {
			familyConfig = {
				...shared,
				caching: { ...shared.caching, ltx2_mode: 'av' },
				training: { ...shared.training, ltx2_mode: 'av' },
				inference: { ...shared.inference, ltx2_mode: 'av' },
			};
		} else if (loraFamily === 'audio') {
			familyConfig = {
				...shared,
				caching: { ...shared.caching, ltx2_mode: 'audio' },
				training: { ...shared.training, ltx2_mode: 'audio', lora_target_preset: 'audio' },
				inference: { ...shared.inference, ltx2_mode: 'audio' },
			};
		} else if (loraFamily === 'ic') {
			familyConfig = {
				...shared,
				caching: { ...shared.caching, reference_frames: 1, reference_downscale: 1 },
				training: {
					...shared.training,
					lora_target_preset: 'v2v',
					ic_lora_strategy: 'v2v',
				},
				inference: { ...shared.inference, sample_include_reference: true },
			};
		} else if (loraFamily === 'slider') {
			familyConfig = {
				...shared,
				slider: {
					...shared.slider,
					mode: 'text',
					output_name: 'ltx2_slider',
					max_train_steps: 500,
				},
			};
		}

		if (memoryProfile === 'high') {
			return {
				...familyConfig,
				caching: {
					...familyConfig.caching,
					gemma_load_in_8bit: false,
					gemma_load_in_4bit: false,
				},
				training: {
					...familyConfig.training,
					gradient_checkpointing: true,
					fp8_base: true,
					fp8_scaled: true,
					gemma_load_in_8bit: false,
					gemma_load_in_4bit: false,
				},
				inference: {
					...familyConfig.inference,
					fp8_base: false,
					gemma_load_in_8bit: false,
					gemma_load_in_4bit: false,
					offloading: false,
				},
			};
		}

		if (memoryProfile === 'low') {
			return {
				...familyConfig,
				caching: {
					...familyConfig.caching,
					gemma_load_in_8bit: false,
					gemma_load_in_4bit: true,
				},
				training: {
					...familyConfig.training,
					gradient_checkpointing: true,
					fp8_base: true,
					fp8_scaled: true,
					gemma_load_in_8bit: false,
					gemma_load_in_4bit: true,
				},
				inference: {
					...familyConfig.inference,
					fp8_base: true,
					gemma_load_in_8bit: false,
					gemma_load_in_4bit: true,
					offloading: true,
				},
			};
		}

		return {
			...familyConfig,
			caching: {
				...familyConfig.caching,
				gemma_load_in_8bit: true,
				gemma_load_in_4bit: false,
			},
			training: {
				...familyConfig.training,
				gradient_checkpointing: true,
				fp8_base: true,
				fp8_scaled: true,
				gemma_load_in_8bit: true,
				gemma_load_in_4bit: false,
			},
			inference: {
				...familyConfig.inference,
				fp8_base: false,
				gemma_load_in_8bit: true,
				gemma_load_in_4bit: false,
				offloading: false,
			},
		};
	}

	async function handleCreate() {
		error = '';
		if (!newProjectName.trim()) {
			error = 'Project name is required';
			return;
		}
		if (projectDirExists) {
			error = `Project already exists: ${newProjectDir}`;
			return;
		}
		creating = true;
		try {
			const modelDir = defaultModelDir(cwd, null);
			const seedConfig = { model_dir: modelDir };
			const defaultLtx = effectiveLtx2Checkpoint(cwd, seedConfig, '');
			const defaultGemma = effectiveGemmaRoot(cwd, seedConfig, '', '');
			const templateConfig = buildTemplateConfig(selectedLoraFamily, selectedMemoryProfile, defaultLtx, defaultGemma);
			await createProject({
				name: newProjectName,
				project_dir: newProjectDir,
				model_dir: modelDir,
				...templateConfig
			});
		} catch (e) { error = e.message; }
		creating = false;
	}

	async function handleLoad() {
		error = '';
		try { await loadProjectFromPath(loadPath); } catch (e) { error = e.message; }
	}

	function handleRemoveRecentProject(project, index) {
		removeRecentProject(project.path);
		removedRecentProjects = [
			{ ...project, removedIndex: index, removedAt: Date.now() },
			...removedRecentProjects.filter((item) => item.path !== project.path)
		].slice(0, 5);
	}

	function handleRestoreRecentProject(project) {
		restoreRecentProject(project, project.removedIndex ?? 0);
		removedRecentProjects = removedRecentProjects.filter((item) => item.path !== project.path);
	}

	async function loadRecentProjectSummary(path) {
		recentProjectSummaryLoading = { ...recentProjectSummaryLoading, [path]: true };
		try {
			const res = await fetch('/api/project/summary', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ path })
			});
			const data = await res.json();
			if (!res.ok) throw new Error(data.detail || 'Failed to read summary');
			recentProjectSummaries = { ...recentProjectSummaries, [path]: data.summary };
		} catch (e) {
			recentProjectSummaries = { ...recentProjectSummaries, [path]: { error: e.message || 'Summary unavailable' } };
		} finally {
			recentProjectSummaryLoading = { ...recentProjectSummaryLoading, [path]: false };
		}
	}

	function compactDatasetSummary(summary) {
		if (!summary) return '';
		const total = summary.datasets ?? 0;
		const val = summary.validation_datasets ?? 0;
		return `${total} ds${val ? ` +${val} val` : ''}`;
	}

	function trainingLengthSummary(summary) {
		if (!summary) return '';
		if (summary.max_train_epochs) return `${summary.max_train_epochs} epochs`;
		return `${summary.max_train_steps || 0} steps`;
	}

	function rankSummary(summary) {
		if (!summary) return '';
		return summary.network_dim ? `rank ${summary.network_dim}` : 'rank auto';
	}

	function projectFolderPath(path) {
		return (path || '').replace(/[\\/][^\\/]*\.json$/i, '');
	}

	function updateConfig(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, [key]: value };
		});
		saveProjectDebounced();
	}

	function readCacheThenTrainSession() {
		try {
			return typeof sessionStorage !== 'undefined' && sessionStorage.getItem(CACHE_THEN_TRAIN_SESSION_KEY) === '1';
		} catch {
			return false;
		}
	}

	function setCacheThenTrainEngaged(value) {
		cacheThenTrainEngaged = value;
		try {
			if (typeof sessionStorage === 'undefined') return;
			if (value) {
				sessionStorage.setItem(CACHE_THEN_TRAIN_SESSION_KEY, '1');
			} else {
				sessionStorage.removeItem(CACHE_THEN_TRAIN_SESSION_KEY);
			}
		} catch {}
	}

	function activeWorkflowProcess(statuses, preferredStage = null) {
		if (preferredStage?.processType && isProcessActive(statuses?.[preferredStage.processType])) {
			return preferredStage;
		}
		return CACHE_THEN_TRAIN_STAGES.find((stage) => stage.processType && isProcessActive(statuses?.[stage.processType])) || null;
	}

	function hasActiveWorkflowProcess(statuses) {
		return !!activeWorkflowProcess(statuses);
	}

	class CacheThenTrainCancelled extends Error {
		constructor(message) {
			super(message);
			this.name = 'CacheThenTrainCancelled';
		}
	}

	const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

	async function waitForProcessSuccess(type, label) {
		let missingPolls = 0;
		let idlePolls = 0;

		while (true) {
			if (cacheThenTrainStopRequested) throw new CacheThenTrainCancelled(`${label} was stopped.`);

			const statuses = await refreshStatuses();
			const processStatus = statuses?.[type];

			if (!processStatus) {
				missingPolls += 1;
				if (missingPolls >= 20) throw new Error(`Lost status while waiting for ${label}`);
				await sleep(1500);
				continue;
			}

			missingPolls = 0;
			const state = processStatus.state || 'idle';
			const exitCode = processStatus.exit_code ?? null;

			if (processStatus.stop_requested) throw new CacheThenTrainCancelled(`${label} was stopped.`);
			if (state === 'finished' && exitCode === 0) return processStatus;
			if (state === 'error' || (exitCode !== null && exitCode !== 0)) {
				throw new Error(`${label} failed${exitCode !== null ? ` with exit code ${exitCode}` : ''}`);
			}

			if (state === 'idle') {
				idlePolls += 1;
				if (idlePolls >= 6) throw new Error(`${label} did not start`);
			} else {
				idlePolls = 0;
			}

			await sleep(1500);
		}
	}

	async function cacheThenTrain() {
		if (cacheThenTrainDisabled) return;

		cacheThenTrainRunning = true;
		setCacheThenTrainEngaged(true);
		cacheThenTrainError = '';
		cacheThenTrainPhase = 'dataset';
		cacheThenTrainStopRequested = false;
		cacheThenTrainStopPending = false;

		try {
			cacheThenTrainMessage = 'Preparing dataset...';
			const prepared = prepareCacheThenTrainConfig($projectConfig);
			if (prepared.changed) {
				projectConfig.set(prepared.config);
				await saveProjectNow();
			}

			if (cacheThenTrainStopRequested) throw new CacheThenTrainCancelled('Workflow was stopped.');
			cacheThenTrainPhase = 'latents';
			cacheThenTrainMessage = 'Caching latents...';
			await startProcess('cache_latents');
			await waitForProcessSuccess('cache_latents', 'Latent caching');
			if (cacheThenTrainStopRequested) throw new CacheThenTrainCancelled('Workflow was stopped.');

			cacheThenTrainPhase = 'text';
			cacheThenTrainMessage = 'Caching text...';
			await startProcess('cache_text');
			await waitForProcessSuccess('cache_text', 'Text caching');
			if (cacheThenTrainStopRequested) throw new CacheThenTrainCancelled('Workflow was stopped.');

			cacheThenTrainPhase = 'training';
			cacheThenTrainMessage = 'Starting training...';
			await startProcess('training');
		} catch (e) {
			cacheThenTrainError = e instanceof CacheThenTrainCancelled ? '' : (e?.message || 'Cache & Start Training failed');
		} finally {
			cacheThenTrainRunning = false;
			cacheThenTrainMessage = '';
			cacheThenTrainPhase = 'idle';
			cacheThenTrainStopPending = false;
		}
	}

	async function stopCacheThenTrain() {
		if (cacheThenTrainStopPending) return;
		cacheThenTrainStopRequested = true;
		cacheThenTrainStopPending = true;
		cacheThenTrainError = '';

		try {
			const statuses = (await refreshStatuses()) || $processStatuses;
			const activeStage = activeWorkflowProcess(statuses, cacheThenTrainProgress.activeStage);
			const processType = activeStage?.processType;
			if (processType && isProcessActive(statuses?.[processType])) {
				cacheThenTrainMessage = `Stopping ${activeStage.label.toLowerCase()}...`;
				await stopProcess(processType);
			} else {
				cacheThenTrainMessage = 'Stopping workflow...';
			}
			await refreshStatuses();
		} catch (e) {
			cacheThenTrainError = e?.message || 'Failed to stop workflow';
		} finally {
			cacheThenTrainStopPending = false;
		}
	}

	async function loadWorkflowStats() {
		const expectedSnapshot = workflowStatsSnapshot;
		try {
			const res = await fetch('/api/stats');
			if (expectedSnapshot !== workflowStatsSnapshot) return;
			workflowStats = res.ok ? await res.json() : null;
		} catch {
			if (expectedSnapshot === workflowStatsSnapshot) workflowStats = null;
		} finally {
			if (expectedSnapshot === workflowStatsSnapshot) workflowStatsLoading = false;
		}
	}

	// VRAM estimation helpers
	function gemmaSize(cfg) {
		// Gemma 3 12B: ~24GB fp16, ~12GB 8bit, ~6GB 4bit
		if (cfg?.gemma_load_in_4bit) return 6;
		if (cfg?.gemma_load_in_8bit) return 12;
		return 24;
	}

	function vaeSize(cfg) {
		// LTX-2 VAE: ~1.5GB bf16/fp16, ~3GB fp32
		const dtype = cfg?.vae_dtype || 'bfloat16';
		return dtype === 'float32' ? 3.0 : 1.5;
	}

	// Three separate VRAM estimations
	function estimateLatentCaching(cfg) {
		if (!cfg?.caching) return null;
		const c = cfg.caching;
		const vae = vaeSize(c);
		// VAE activation memory scales with input resolution and frames.
		// Base: ~2.5 GB for 512x768x33f. Scales roughly proportional to pixel*frame count.
		const allDatasets = cfg?.dataset?.datasets || [];
		const ds = allDatasets.find((d) => d?.type === 'video' || d?.type === 'image') || allDatasets[0] || {};
		const resW = Math.max(Number(ds.resolution_w || 768), 64);
		const resH = Math.max(Number(ds.resolution_h || 512), 64);
		const frames = Math.max(Number(ds.target_frames || 33), 1);
		const basePixelFrames = 512 * 768 * 33;
		const pixelFrames = resW * resH * frames;
		const resScale = Math.max(0.5, Math.min(pixelFrames / basePixelFrames, 4.0));
		// VAE tiling reduces activation memory significantly
		const hasSpatialTiling = !!(c.vae_spatial_tile_size || c.vae_chunk_size);
		const hasTemporalTiling = !!c.vae_temporal_tile_size;
		const tilingFactor = (hasSpatialTiling && hasTemporalTiling) ? 0.2 :
			hasSpatialTiling ? 0.3 : hasTemporalTiling ? 0.5 : 1.0;
		const buffer = 2.5 * resScale * tilingFactor;
		const total = vae + buffer;
		return {
			total: Math.max(total, 1),
			parts: [
				{ label: 'VAE', value: vae, color: 'var(--accent)' },
				{ label: 'Activations', value: buffer, color: 'var(--info)' },
			]
		};
	}

	function estimateTextCaching(cfg) {
		if (!cfg?.caching) return null;
		const c = cfg.caching;
		const gemma = gemmaSize(c);
		// Buffer for embeddings and intermediate tensors
		const buffer = 2.0;
		const total = gemma + buffer;
		return {
			total: Math.max(total, 1),
			parts: [
				{ label: 'Gemma', value: gemma, color: 'var(--accent)' },
				{ label: 'Buffer', value: buffer, color: 'var(--info)' },
			]
		};
	}

	function estimateTraining(cfg) {
		if (!cfg?.training) return null;
		const t = cfg.training;
		const allDatasets = cfg?.dataset?.datasets || [];
		const ds = allDatasets.find((d) => d?.type === 'video' || d?.type === 'image') || allDatasets[0] || {};

		// ── DiT weights ──
		// LTX-2 DiT: 48 transformer blocks. VAE/Gemma NOT resident during training.
		// LTX 2.0: ~19.6B params → BF16 39 GB, FP8 19.5 GB, FP32 78 GB
		// LTX 2.3: ~21.0B params → BF16 42 GB, FP8 21 GB, FP32 84 GB
		const ltxVersion = String(t.ltx_version || '2.3');
		const ditBF16 = ltxVersion === '2.3' ? 42 : 39;
		const isFp8 = !!t.fp8_base;
		const isW8A8 = !!t.fp8_w8a8;
		const isNF4 = !!t.nf4_base;
		let ditBase = isNF4 ? (ditBF16 / 4) : isFp8 ? (ditBF16 / 2) : ditBF16;

		const totalBlocks = 48;
		const blocksToSwap = Math.min(Math.max(Number(t.blocks_to_swap || 0), 0), totalBlocks - 1);
		const blockSize = ditBase / totalBlocks;
		const swapSavings = blocksToSwap * blockSize * 0.95;
		const dit = Math.max(ditBase - swapSavings, 1.0);

		// ── LoRA weights ──
		// Per target linear: rank * (in_features + out_features) * 2 bytes (bf16).
		// t2v (attn Q/K/V/Out): 8 linears/block * Linear(4096,4096) = rank * 131,072 bytes/block
		//   + audio attn: rank * 65,536 bytes/block + cross-modal: rank * 81,920 bytes/block
		//   48 blocks -> rank * 12.75 MB total (video-only ~rank * 6.0 MB)
		// v2v adds FFN: +rank * 5.9 MB. full adds all remaining: +rank * ~8 MB.
		const rank = Math.max(Number(t.network_dim || 16), 1);
		const mode = String(t.ltx2_mode || 'video');
		const isAV = mode === 'av';
		// Base LoRA size in GB per unit rank (t2v preset, video-only)
		const loraBasePerRank = isAV ? 12.75 / 1024 : 6.0 / 1024;  // GB per rank
		const presetMultiplier = { t2v: 1.0, v2v: 1.44, video_sa: 0.37, video_sa_ff: 0.56, video_sa_ca_ff: 0.74, audio: 0.37, audio_v2a: 0.52, audio_ref_ic: 0.63, av_ic: 1.44, video_ref_only_av: 1.44, full: 2.1 }[t.lora_target_preset] || 1.0;
		const loraParamsGB = rank * loraBasePerRank * presetMultiplier;

		// ── Optimizer states ──
		// AdamW fp32: 12 bytes/param (fp32 master + momentum + variance)
		// AdamW 8-bit: 6 bytes/param (fp32 master + int8 momentum + int8 variance)
		// Prodigy/ScheduleFree: ~14 bytes/param
		const loraParamCount = loraParamsGB * (1024 ** 3) / 2;  // bf16 -> param count
		const optType = String(t.optimizer_type || 'adamw8bit').toLowerCase();
		const is8bitOpt = optType.includes('8bit');
		const isScheduleFree = optType.includes('schedulefree') || optType === 'automagic';
		const optBytesPerParam = is8bitOpt ? 6 : (isScheduleFree ? 14 : 12);
		const optimStates = (loraParamCount * optBytesPerParam) / (1024 ** 3);

		// ── Gradients ──
		// One gradient per trainable param in training precision (bf16 = 2 bytes)
		const loraGrads = loraParamsGB;  // same size as LoRA weights in bf16

		// ── Activations ──
		// LTX-2 VAE compression: temporal 8x, spatial 32x32.
		// Latent shape: (B, 128, 1+(F-1)/8, H/32, W/32)
		// Sequence length = latent_F * latent_H * latent_W (patch_size=1)
		// DiT hidden_dim: 4096 (video), 2048 (audio)
		const resolutionW = Math.max(Number(ds.resolution_w || 768), 64);
		const resolutionH = Math.max(Number(ds.resolution_h || 512), 64);
		const sourceFrames = Math.max(Number(ds.target_frames || 33), 1);
		const batchSize = Math.max(Number(ds.batch_size || 1), 1);

		const latentFrames = Math.max(1, Math.floor((sourceFrames - 1) / 8) + 1);
		const latentHeight = Math.max(1, Math.floor(resolutionH / 32));
		const latentWidth = Math.max(1, Math.floor(resolutionW / 32));
		let seqTokens = latentFrames * latentHeight * latentWidth;

		// Audio adds ~25 tokens per second of video (at 25fps)
		const audioTokens = isAV ? Math.round(sourceFrames) : 0;
		if (mode === 'audio') seqTokens = Math.round(sourceFrames);  // audio-only

		const hiddenDim = mode === 'audio' ? 2048 : 4096;
		const bytesPerValue = isW8A8 ? 1 : 2;

		// Per-block activation: ~10 tensors of (batch, seq_len, hidden_dim) without checkpointing,
		// ~2 with gradient checkpointing (only block boundaries stored, recomputed in backward).
		// With blockwise checkpointing: ~1 (activations offloaded to CPU).
		const activCoeff = (t.gradient_checkpointing === false) ? 10 : (t.blockwise_checkpointing ? 1 : 2);

		// Effective stored layers: without checkpointing all 48; with checkpointing, 48 boundaries
		// are stored but each is small (~1 tensor). Plus one block's full activations during recompute.
		const effectiveLayers = (t.gradient_checkpointing === false) ? totalBlocks : (t.blockwise_checkpointing ? 2 : totalBlocks);

		const perLayerBytes = activCoeff * batchSize * seqTokens * hiddenDim * bytesPerValue;
		let activations = (perLayerBytes * effectiveLayers) / (1024 ** 3);

		// Audio stream adds ~25% activation overhead in AV mode (hidden_dim=2048, separate path)
		if (isAV) activations *= 1.25;

		// Memory-saving techniques
		if ((t.ffn_chunk_size || 0) > 0) activations *= 0.90;
		if (t.split_attn_mode || t.split_attn_target) activations *= 0.92;
		// GC CPU offload: activation checkpoints stored on CPU instead of GPU
		if (t.gradient_checkpointing_cpu_offload && t.gradient_checkpointing !== false) activations *= 0.35;

		// Fixed buffers: CUDA allocator overhead, latent tensors, noise, text embeddings
		// Latents: batch * 128 * latentF * latentH * latentW * 2 bytes (small, ~tens of MB)
		const latentBytes = batchSize * 128 * latentFrames * latentHeight * latentWidth * 2 * 2; // x2 for noise
		const textEmbedBytes = batchSize * 256 * (isAV ? 7680 : 3840) * 2;  // 256 text tokens, caption_channels
		const bufferGB = (latentBytes + textEmbedBytes) / (1024 ** 3);
		let activationBuffers = 0.5 + bufferGB;  // 0.5 GB base CUDA allocator overhead
		if (t.img_in_txt_in_offloading) activationBuffers = Math.max(0.2, activationBuffers - 0.3);

		const activationTotal = Math.max(0.3, activations + activationBuffers);

		// ── Gradient accumulation ──
		const gradAccum = Math.max(Number(t.gradient_accumulation_steps || 1), 1);
		const gradAccumOverhead = gradAccum > 1 ? loraGrads * 0.4 : 0;

		// ── Preservation / DOP ──
		let preservationOverhead = 0;
		if (t.blank_preservation) preservationOverhead += activationTotal * 0.35;
		if (t.dop) preservationOverhead += activationTotal * 0.35;
		if (t.audio_dop) preservationOverhead += activationTotal * 0.35;
		if (t.prior_divergence) preservationOverhead += activationTotal * 0.15;

		// ── Self-Flow ──
		// teacher_mode=base reuses the frozen base via zeroed LoRA multipliers.
		// EMA modes keep shadow trainable params; offload_teacher_features only moves cached features.
		let selfFlowOverhead = 0;
		if (t.self_flow) {
			const teacherMode = String(t.self_flow_teacher_mode || 'base').toLowerCase();
			if (teacherMode === 'ema') selfFlowOverhead += loraParamsGB;
			else if (teacherMode === 'partial_ema') selfFlowOverhead += Math.max(loraParamsGB / totalBlocks, 0.01);

			const hasAudioProjector = (mode === 'av' || mode === 'audio') && Number(t.self_flow_lambda_audio || 0) > 0;
			selfFlowOverhead += hasAudioProjector ? 0.03 : 0.02;
			selfFlowOverhead += activationTotal * (t.self_flow_offload_teacher_features ? 0.03 : 0.10);
		}

		// ── CREPA ──
		const crepaOverhead = t.crepa ? (String(t.crepa_mode || 'backbone') === 'dino' ? 0.08 : 0.15) : 0;

		const total = dit + loraParamsGB + optimStates + loraGrads + activationTotal + gradAccumOverhead + preservationOverhead + selfFlowOverhead + crepaOverhead;

		// Temporary spikes (not steady-state)
		const hasSamplePrompts = !!(t.sample_prompts || t.sample_prompts_text);
		const samplingEnabled = hasSamplePrompts && !!(t.sample_at_first || t.sample_every_n_steps || t.sample_every_n_epochs);
		const samplingSpike = samplingEnabled;
		const preservationGemmaSpike = !!(t.blank_preservation || t.dop || t.audio_dop) && !t.use_precached_preservation;

		const parts = [
			{ label: 'DiT', value: dit, color: 'var(--accent)' },
		];
		parts.push({ label: 'LoRA', value: loraParamsGB, color: 'var(--warning)' });
		parts.push({ label: 'Optimizer', value: optimStates, color: 'var(--warning)' });
		parts.push({ label: 'Grads', value: loraGrads, color: 'var(--info)' });
		parts.push({ label: 'Activ.', value: activationTotal, color: 'var(--success)' });
		if (gradAccumOverhead > 0) parts.push({ label: 'GradAccum', value: gradAccumOverhead, color: 'var(--info)' });
		if (preservationOverhead > 0) parts.push({ label: 'Preserv.', value: preservationOverhead, color: 'var(--danger)' });
		if (selfFlowOverhead > 0) parts.push({ label: 'Self-Flow', value: selfFlowOverhead, color: 'var(--danger)' });
		if (crepaOverhead > 0) parts.push({ label: 'CREPA', value: crepaOverhead, color: 'var(--secondary, var(--info))' });

		return {
			total: Math.max(total, 2),
			parts,
			swap: swapSavings,
			noGemma: true,
			samplingSpike,
			preservationGemmaSpike,
		};
	}

	let cfg = $derived($projectConfig);
	let datasets = $derived(cfg?.dataset?.datasets || []);
	let valDatasets = $derived(cfg?.dataset?.validation_datasets || []);
	let t = $derived(cfg?.training || {});
	let c = $derived(cfg?.caching || {});
	let vramLatent = $derived(estimateLatentCachingVram(cfg));
	let vramText = $derived(estimateTextCachingVram(cfg));
	let vramTrain = $derived(estimateTrainingVram(cfg));
	let vramTotal = $derived(systemInfo?.gpus?.[0] ? Math.round(systemInfo.gpus[0].vram_total_mb / 1024) : 24);
	let gpu = $derived(systemInfo?.gpus?.[0]);
	let cacheThenTrainDisabled = $derived(
		cacheThenTrainRunning ||
		isProcessActive($processStatuses.cache_latents) ||
		isProcessActive($processStatuses.cache_text) ||
		isProcessActive($processStatuses.training)
	);
	let cacheThenTrainProgress = $derived.by(() => buildCacheThenTrainProgress({
		statuses: $processStatuses,
		logs: $processLogs,
		trainingStatus: $status,
		config: cfg,
		fallbackStageId: cacheThenTrainPhase,
		fallbackMessage: cacheThenTrainMessage,
		workflowRunning: cacheThenTrainRunning,
	}));
	let cacheThenTrainProcessActive = $derived(
		cacheThenTrainRunning ||
		CACHE_THEN_TRAIN_STAGES.some((stage) => stage.processType && isProcessActive($processStatuses[stage.processType]))
	);
	let cacheThenTrainWorkflowActive = $derived(
		cacheThenTrainEngaged ||
		cacheThenTrainRunning ||
		cacheThenTrainProcessActive ||
		cacheThenTrainPhase !== 'idle' ||
		cacheThenTrainStopPending
	);
	let cacheThenTrainCanStop = $derived(
		(cacheThenTrainRunning || cacheThenTrainProcessActive) &&
		!cacheThenTrainStopPending
	);
	let cacheThenTrainProgressVisible = $derived(cacheThenTrainWorkflowActive);
	let cacheThenTrainIdleSummary = $derived(
		workflowStatsLoading
			? {
				label: 'Estimating Runtime',
				detail: 'Reading dataset and training settings',
				metrics: [],
			}
			: buildCacheThenTrainIdleSummary(cfg, workflowStats)
	);

	$effect(() => {
		if (!$projectLoaded || !cfg) {
			workflowStats = null;
			workflowStatsLoading = false;
			workflowStatsSnapshot = '';
			if (workflowStatsTimer) clearTimeout(workflowStatsTimer);
			return;
		}

		const snapshot = JSON.stringify({
			dataset: cfg.dataset,
			caching: {
				ltx2_mode: cfg.caching?.ltx2_mode,
			},
			training: {
				ltx2_mode: cfg.training?.ltx2_mode,
				max_train_steps: cfg.training?.max_train_steps,
				gradient_accumulation_steps: cfg.training?.gradient_accumulation_steps,
				network_dim: cfg.training?.network_dim,
				lora_target_preset: cfg.training?.lora_target_preset,
				fp8_base: cfg.training?.fp8_base,
			},
		});
		if (snapshot === workflowStatsSnapshot) return;

		workflowStatsSnapshot = snapshot;
		if (workflowStatsTimer) clearTimeout(workflowStatsTimer);
		workflowStatsLoading = true;
		workflowStatsTimer = setTimeout(loadWorkflowStats, 900);
	});

	// Training progress from status store
	let trainingProgress = $derived.by(() => {
		const s = $status;
		const ts = $processStatuses.training;
		if (!ts || ts.state !== 'running') return null;
		if (!s) return { running: true };
		return {
			running: true,
			step: s.step || 0,
			maxSteps: s.max_steps || 0,
			epoch: s.epoch || 0,
			loss: s.loss != null ? s.loss.toFixed(4) : null,
			lossV: s.loss_v != null ? s.loss_v.toFixed(4) : null,
			lossA: s.loss_a != null ? s.loss_a.toFixed(4) : null,
			speed: s.speed_steps_per_sec ? s.speed_steps_per_sec.toFixed(2) : null,
		};
	});

	// Diagnostics: detect blockers/errors for each stage
	let diagnostics = $derived.by(() => {
		const errors = [];
		const warnings = [];
		const hasLtxPath = !!(c.ltx2_checkpoint || t.ltx2_checkpoint || cfg?.default_ltx2_checkpoint || cfg?.model_dir || cwd);
		const hasGemmaPath = !!(c.gemma_root || t.gemma_root || cfg?.default_gemma_root || cfg?.default_gemma_safetensors || cfg?.model_dir || cwd);

		// Dataset
		if (datasets.length === 0) {
			errors.push({ stage: 'Dataset', msg: 'No datasets configured', href: '/dataset' });
		} else {
			const emptyDirs = datasets.filter(ds => !ds.directory);
			if (emptyDirs.length > 0) {
				errors.push({ stage: 'Dataset', msg: `${emptyDirs.length} dataset(s) missing directory path`, href: '/dataset' });
			}
		}

		// Caching
		if (!hasLtxPath) {
			errors.push({ stage: 'Caching', msg: 'LTX-2 checkpoint path not set', href: '/caching' });
		}
		if (!hasGemmaPath) {
			errors.push({ stage: 'Caching', msg: 'Gemma text encoder path not set', href: '/caching' });
		}

		// Training
		if (!hasLtxPath) {
			warnings.push({ stage: 'Training', msg: 'Training checkpoint not set', href: '/training' });
		}
		if (!hasGemmaPath) {
			warnings.push({ stage: 'Training', msg: 'Training Gemma root not set', href: '/training' });
		}
		if (!t.output_dir) {
			errors.push({ stage: 'Training', msg: 'Output directory not set', href: '/training' });
		}

		return { errors, warnings, total: errors.length + warnings.length };
	});
</script>

{#if !$projectLoaded}
	<div class="h-full max-w-7xl mx-auto px-4 sm:px-6 flex flex-col gap-3 min-h-0">
		<!-- Header -->
		<div class="flex flex-wrap items-center justify-between gap-3">
			<div class="flex items-center gap-2.5 min-w-0">
				<div class="w-7 h-7 flex items-center justify-center flex-shrink-0" style="background: var(--logo-bg); box-shadow: var(--logo-shadow); border-radius: var(--logo-radius); clip-path: var(--logo-clip);">
					<span class="text-[11px] font-bold" style="color: var(--bg-base);">M</span>
				</div>
				<div class="flex flex-wrap items-baseline gap-x-2 min-w-0">
					<h2 class="text-base font-semibold" style="color: var(--text-primary);">Musubi Tuner</h2>
					<span class="text-[12px]" style="color: var(--text-muted);">LTX-2 LoRA training management</span>
				</div>
			</div>
			<div class="flex flex-wrap items-center gap-2 text-[11px]">
				<a
					href={ltxDocsUrl}
					target="_blank"
					rel="noreferrer"
					class="title-docs-link px-2 py-1 font-medium"
					style="background: var(--bg-elevated); color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 26%, var(--border)); border-radius: var(--radius-sm);"
				>
					LTX-2 docs
				</a>
				<span class="font-mono text-[10px] px-1.5 py-0.5" style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);">{ltxDocsBranch}</span>
			</div>
		</div>

		<!-- Two-panel layout -->
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-5 min-h-0 flex-1">
			<!-- Left: Create New Project -->
			<div class="p-5 space-y-4 min-h-0" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
				<div class="flex items-center justify-between gap-4">
					<div>
						<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">New Project</div>
					</div>
					<div class="text-[11px]" style="color: var(--text-muted);">{activeLoraFamily.label} • {activeMemoryProfile.label}</div>
				</div>
				<div class="space-y-3">
					<div class="project-name-field">
						<FormField label="Project Name" bind:value={newProjectName} placeholder="My LTX-2 LoRA" tooltip="Display name for your project" invalid={projectDirExists} />
						<div class="project-name-status">
							{#if projectDirExists}
								<span style="color: var(--danger);">Name exists</span>
							{:else if projectDirCheckPending}
								<span style="color: var(--text-muted);">Checking...</span>
							{/if}
						</div>
					</div>
					<div class="project-path-preview" data-tooltip={newProjectDir || undefined}>
						<span>Folder</span>
						<strong>{newProjectDir || 'Waiting for repository path...'}</strong>
					</div>

					<div class="space-y-2">
						<div class="flex items-center justify-between gap-3">
							<div class="text-[11px] font-medium uppercase tracking-[0.18em]" style="color: var(--text-secondary); font-family: var(--font-label);">LoRA Type</div>
							<div class="text-[10px]" style="color: var(--text-muted);">{activeLoraFamily.label}</div>
						</div>
						<div class="flex flex-wrap gap-2">
							{#each LORA_FAMILIES as family}
								<button
									type="button"
									disabled={TEMPLATE_VARIANTS_DISABLED}
									onclick={() => {
										if (!TEMPLATE_VARIANTS_DISABLED) selectedLoraFamily = family.id;
									}}
									class="choice-pill"
									style="background: {selectedLoraFamily === family.id ? 'color-mix(in srgb, var(--accent) 10%, var(--bg-elevated))' : 'var(--bg-elevated)'}; border: 1px solid {selectedLoraFamily === family.id ? 'color-mix(in srgb, var(--accent) 28%, var(--border))' : 'var(--border)'}; color: {selectedLoraFamily === family.id ? 'var(--text-primary)' : 'var(--text-secondary)'};"
								>
									{family.label}
								</button>
							{/each}
						</div>
						<div class="text-[11px]" style="color: var(--text-muted);">{activeLoraFamily.blurb}</div>
					</div>

					<div class="space-y-2">
						<div class="flex items-center justify-between gap-3">
							<div class="text-[11px] font-medium uppercase tracking-[0.18em]" style="color: var(--text-secondary); font-family: var(--font-label);">Memory Profile</div>
							<div class="text-[10px]" style="color: var(--text-muted);">{activeMemoryProfile.label}</div>
						</div>
						<div class="flex flex-wrap gap-2">
							{#each MEMORY_PROFILES as profile}
								<button
									type="button"
									disabled={TEMPLATE_VARIANTS_DISABLED}
									onclick={() => {
										if (!TEMPLATE_VARIANTS_DISABLED) selectedMemoryProfile = profile.id;
									}}
									class="choice-pill"
									style="background: {selectedMemoryProfile === profile.id ? 'color-mix(in srgb, var(--accent) 10%, var(--bg-elevated))' : 'var(--bg-elevated)'}; border: 1px solid {selectedMemoryProfile === profile.id ? 'color-mix(in srgb, var(--accent) 28%, var(--border))' : 'var(--border)'}; color: {selectedMemoryProfile === profile.id ? 'var(--text-primary)' : 'var(--text-secondary)'};"
								>
									{profile.label}
								</button>
							{/each}
						</div>
						<div class="text-[11px]" style="color: var(--text-muted);">{activeMemoryProfile.blurb}</div>
					</div>

					<button
						onclick={handleCreate}
						disabled={!newProjectDir || projectDirExists || projectDirCheckPending || creating}
						class="w-full py-2.5 text-[13px] font-semibold disabled:opacity-40"
						style="background: color-mix(in srgb, var(--accent) 74%, var(--bg-elevated)); color: var(--text-primary); border: 1px solid color-mix(in srgb, var(--accent) 32%, var(--border)); border-radius: var(--radius-sm);"
					>{creating ? 'Creating...' : 'Create Project'}</button>
				</div>
			</div>

			<!-- Right: Recent Projects + Load Existing -->
			<div class="p-5 min-h-0 flex flex-col gap-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Open Project</div>

				{#if $recentProjects.length > 0}
					<div class="space-y-3 min-h-0 flex-1 flex flex-col">
						<div class="flex items-center justify-between">
							<p class="text-[11px] font-medium uppercase tracking-[0.18em]" style="color: var(--text-secondary); font-family: var(--font-label);">Recent</p>
							<span class="text-[10px] px-2 py-0.5 rounded-full" style="background: var(--bg-elevated); color: var(--text-muted); border: 1px solid var(--border);">{$recentProjects.length}</span>
						</div>
						<div class="recent-project-list flex-1 min-h-0 space-y-1.5">
						{#each $recentProjects as proj, index}
							{@const summary = recentProjectSummaries[proj.path]}
							{@const summaryLoading = recentProjectSummaryLoading[proj.path]}
							<div class="recent-project-row">
								<div class="min-w-0 flex-1">
									<div class="flex items-center min-w-0">
										<div class="min-w-0 flex-1">
											<div class="text-[13px] font-semibold truncate" style="color: var(--text-primary);" data-tooltip={projectFolderPath(proj.path)}>{proj.name}</div>
										</div>
									</div>
									<div class="mt-1.5 min-w-0">
										{#if summary?.error}
											<div class="text-[11px] truncate" style="color: var(--danger);">{summary.error}</div>
										{:else if summary}
											<div class="recent-project-meta">
												<span class="summary-dataset">{compactDatasetSummary(summary)}</span>
												<span class="summary-version">LTX {summary.ltx_version}</span>
												<span class="summary-mode">{summary.mode}</span>
												<span class="summary-target">{summary.lora_target_preset}</span>
												{#if summary.lora_kind !== 'LoRA'}
													<span class="summary-kind">{summary.lora_kind}</span>
												{/if}
												<span class="summary-output">{summary.output_name || 'ltx2_lora'}</span>
												<span class="summary-steps">{trainingLengthSummary(summary)}</span>
												<span class="summary-rank">{rankSummary(summary)}</span>
											</div>
										{:else if summaryLoading}
											<div class="text-[11px] truncate" style="color: var(--text-muted);">Reading summary...</div>
										{/if}
									</div>
								</div>
								<div class="recent-project-actions">
									<button
										type="button"
										onclick={() => { loadPath = proj.path; handleLoad(); }}
										class="recent-project-action recent-project-action-primary px-3 py-1.5 text-[11px] font-medium"
									>
										Open
									</button>
									<button
										type="button"
										onclick={() => handleRemoveRecentProject(proj, index)}
										class="recent-project-action recent-project-action-muted px-2.5 py-1.5 text-[11px] font-medium"
									>
										Remove
									</button>
								</div>
							</div>
						{/each}
						</div>
					</div>
				{/if}

				{#if removedRecentProjects.length > 0}
					<div class="space-y-2.5" style="padding-top: 0.5rem; border-top: 1px solid var(--border-subtle);">
						<div class="flex items-center justify-between">
							<p class="text-[11px] font-medium uppercase tracking-[0.18em]" style="color: var(--text-secondary); font-family: var(--font-label);">Removed</p>
							<span class="text-[10px] px-2 py-0.5 rounded-full" style="background: var(--bg-elevated); color: var(--text-muted); border: 1px solid var(--border);">{removedRecentProjects.length}</span>
						</div>
						<div class="space-y-2">
							{#each removedRecentProjects as proj}
								<div class="removed-project-row flex items-center justify-between gap-3 px-3 py-2">
									<div class="min-w-0 flex-1">
										<div class="text-[11px] font-medium truncate" style="color: var(--text-primary);">{proj.name}</div>
										<div class="text-[10px] truncate" style="color: var(--text-muted);">{proj.path}</div>
									</div>
									<button
										type="button"
										onclick={() => handleRestoreRecentProject(proj)}
										class="recent-project-action recent-project-action-primary px-3 py-2 text-[11px] font-medium"
									>
										Restore
									</button>
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<div class="space-y-2" style="{$recentProjects.length > 0 ? 'padding-top: 0.5rem; border-top: 1px solid var(--border-subtle);' : ''}">
					<p class="text-[11px] font-medium" style="color: var(--text-muted);">Load from file</p>
					<PathInput label="Project File" bind:value={loadPath} placeholder="path/to/project.json" showFiles tooltip="Path to an existing project.json file or project directory" />
					<button
						onclick={handleLoad}
						disabled={!loadPath}
						class="w-full py-2 text-[12px] font-medium disabled:opacity-40"
						style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
					>Load Project</button>
				</div>
			</div>
		</div>

		{#if error}
			<div class="text-[12px] px-3 py-2" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">{error}</div>
		{/if}
	</div>
{:else}
	<div class="space-y-4">
		{#if cacheThenTrainError}
			<div class="text-[12px] px-3 py-2" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">{cacheThenTrainError}</div>
		{/if}

		<div class="workflow-panel p-4">
			<div class="flex flex-col xl:flex-row xl:items-start justify-between gap-4">
				<div class="workflow-main min-w-0">
					<div class="flex flex-wrap items-center justify-between gap-3 mb-2">
						<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Project Workflow</div>
					</div>

					<div class="workflow-stages" aria-label="Cache and training stages">
						{#each cacheThenTrainProgress.stages as stage, index}
							<a
								href={stage.href}
								class="workflow-stage"
								class:workflow-stage-active={stage.state === 'active'}
								class:workflow-stage-complete={stage.state === 'complete'}
								aria-current={stage.state === 'active' ? 'step' : undefined}
							>
								<span class="workflow-stage-number">{index + 1}</span>
								<span>{stage.label}</span>
							</a>
							{#if index < cacheThenTrainProgress.stages.length - 1}
								<span class="workflow-stage-arrow">→</span>
							{/if}
						{/each}
					</div>
				</div>

				<div class="workflow-actions">
					{#if cacheThenTrainWorkflowActive && cacheThenTrainProgress.navHref}
						<a href={cacheThenTrainProgress.navHref} class="workflow-secondary-action">
							{cacheThenTrainProgress.navLabel}
						</a>
					{/if}
					{#if cacheThenTrainWorkflowActive}
						<button
							type="button"
							onclick={stopCacheThenTrain}
							disabled={!cacheThenTrainCanStop}
							class="workflow-stop-action disabled:opacity-50 disabled:cursor-not-allowed"
						>
							{cacheThenTrainStopPending ? 'Stopping...' : 'Stop'}
						</button>
					{/if}
					<button
						type="button"
						onclick={cacheThenTrain}
						disabled={cacheThenTrainDisabled}
						class="workflow-primary-action disabled:opacity-50 disabled:cursor-not-allowed"
					>
						{cacheThenTrainWorkflowActive && cacheThenTrainDisabled ? 'Workflow Running' : 'Cache & Start Training'}
					</button>
				</div>
			</div>

			<div
				class="workflow-progress-zone"
				class:workflow-progress-zone-visible={cacheThenTrainProgressVisible}
				aria-live="polite"
			>
				{#if cacheThenTrainProgressVisible}
					<div class="workflow-progress-copy">
						<div class="workflow-progress-text">
							<span class="workflow-progress-label">{cacheThenTrainProgress.label}</span>
							<span class="workflow-progress-detail">{cacheThenTrainProgress.detail}</span>
						</div>
						<strong class="workflow-progress-percent">{Math.round(cacheThenTrainProgress.percent)}%</strong>
					</div>
					<div class="workflow-progress" aria-label="Workflow progress">
						<div class="workflow-progress-fill" style="width: {cacheThenTrainProgress.percent}%;"></div>
					</div>
				{:else}
					<div class="workflow-idle-summary">
						<div class="workflow-idle-copy">
							{#if workflowStatsLoading}
								<span class="workflow-idle-spinner" aria-hidden="true"></span>
							{/if}
							<span class="workflow-progress-label">{cacheThenTrainIdleSummary.label}</span>
							<span class="workflow-progress-detail">{cacheThenTrainIdleSummary.detail}</span>
						</div>
						{#if cacheThenTrainIdleSummary.metrics?.length}
							<div class="workflow-idle-metrics" aria-label="Workflow estimate metrics">
								{#each cacheThenTrainIdleSummary.metrics as metric}
									<div class="workflow-idle-metric">
										<span class="workflow-idle-metric-value">{metric.value}</span>
										<span class="workflow-idle-metric-label">{metric.label}</span>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{/if}
			</div>
		</div>

		<!-- VRAM Estimation -->
		<div class="p-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="flex items-center justify-between gap-3 mb-3">
				<span class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">
					VRAM Estimation{#if gpu} — {gpu.name.replace('NVIDIA ','').replace('GeForce ','')}{/if}
				</span>
				<span class="text-[12px] tabular-nums" style="color: var(--text-muted);">{vramTotal} GB total</span>
			</div>
			<div class="grid grid-cols-1 xl:grid-cols-3 gap-3 items-stretch">
				{#if vramLatent}
					<VramEstimateCard title="Latent Cache" subtitle="VAE encode pass" estimate={vramLatent} total={vramTotal} color="var(--accent)" />
				{/if}
				{#if vramText}
					<VramEstimateCard title="Text Cache" subtitle="Gemma encode pass" estimate={vramText} total={vramTotal} color="var(--info)" />
				{/if}
				{#if vramTrain}
					<VramEstimateCard title="Training" subtitle="DiT + LoRA step" estimate={vramTrain} total={vramTotal} color="var(--warning)" />
				{/if}
			</div>
			{#if vramTrain && vramTrain.total > vramTotal}
				<div class="mt-2 text-[10px] px-2 py-1 flex items-center gap-1.5" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">
					<svg class="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
					Training may exceed VRAM — try FP8, quantization, blocks_to_swap, or lower resolution/frames.
				</div>
			{/if}
		</div>

		<!-- Project Statistics -->
		<StatsPanel />

		<!-- Training Progress (conditional) -->
		{#if trainingProgress}
			<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--accent); border-radius: var(--radius-md);">
				<div class="flex items-center justify-between mb-2">
					<div class="flex items-center gap-2">
						<span class="w-1.5 h-1.5 rounded-full" style="background: var(--success); box-shadow: 0 0 4px var(--success);"></span>
						<span class="text-[11px] font-medium" style="color: var(--text-muted);">Training</span>
					</div>
					<a href="/training/dashboard" class="text-[11px] font-medium" style="color: var(--accent);">Dashboard</a>
				</div>
				{#if trainingProgress.maxSteps > 0}
					<div class="flex items-baseline gap-2 mb-1.5">
						<span class="text-[16px] font-bold tabular-nums" style="color: var(--text-primary);">{trainingProgress.step}</span>
						<span class="text-[11px]" style="color: var(--text-muted);">/ {trainingProgress.maxSteps}</span>
						{#if trainingProgress.epoch}<span class="text-[11px] ml-1" style="color: var(--text-muted);">ep {trainingProgress.epoch}</span>{/if}
					</div>
					<div class="h-1 overflow-hidden mb-2" style="background: var(--border); border-radius: var(--radius-full);">
						<div class="h-full" style="width: {Math.min((trainingProgress.step / trainingProgress.maxSteps) * 100, 100).toFixed(1)}%; background: var(--accent); border-radius: var(--radius-full); transition: width 0.5s ease;"></div>
					</div>
					<div class="flex gap-3 text-[11px]" style="color: var(--text-muted);">
						{#if trainingProgress.loss}<span>loss <span class="tabular-nums" style="color: var(--text-primary);">{trainingProgress.loss}</span></span>{/if}
						{#if trainingProgress.lossV}<span>v <span class="tabular-nums" style="color: var(--text-primary);">{trainingProgress.lossV}</span></span>{/if}
						{#if trainingProgress.lossA}<span>a <span class="tabular-nums" style="color: var(--text-primary);">{trainingProgress.lossA}</span></span>{/if}
						{#if trainingProgress.speed}<span>{trainingProgress.speed} it/s</span>{/if}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Config summary row -->
		<div class="grid grid-cols-4 sm:grid-cols-8 gap-x-3 gap-y-2 px-1">
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Steps</div>
				<div class="text-[13px] font-semibold tabular-nums" style="color: var(--text-primary);">{t.max_train_steps || 0}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">LR</div>
				<div class="text-[13px] font-semibold" style="color: var(--text-primary);">{t.learning_rate || '1e-4'}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Dim</div>
				<div class="text-[13px] font-semibold tabular-nums" style="color: var(--text-primary);">{t.network_dim || 16}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Alpha</div>
				<div class="text-[13px] font-semibold tabular-nums" style="color: var(--text-primary);">{t.network_alpha || 16}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Optimizer</div>
				<div class="text-[13px] font-semibold truncate" style="color: var(--text-primary);">{(t.optimizer_type || 'adamw8bit').replace('adamw8bit','AdamW8')}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Scheduler</div>
				<div class="text-[13px] font-semibold truncate" style="color: var(--text-primary);">{(t.lr_scheduler || 'cosine').replace('constant_with_warmup','const+warm')}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Swap</div>
				<div class="text-[13px] font-semibold tabular-nums" style="color: var(--text-primary);">{t.blocks_to_swap ?? 0}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Target</div>
				<div class="text-[13px] font-semibold" style="color: var(--text-primary);">{t.lora_target_preset || 't2v'}</div>
			</div>
		</div>

		<!-- Datasets -->
		<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="flex items-center justify-between mb-2">
				<span class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Datasets ({datasets.length}{valDatasets.length > 0 ? ` + ${valDatasets.length} val` : ''})</span>
				<a href="/dataset" class="text-[10px] font-medium" style="color: var(--accent);">Edit</a>
			</div>
			{#if datasets.length === 0 && valDatasets.length === 0}
				<div class="text-[11px] py-3 text-center" style="color: var(--text-muted);">No datasets configured</div>
			{:else}
				<div class="space-y-1">
					{#each datasets as ds, i}
						{@const dsType = ds.type || 'video'}
						{@const dsColor = dsType === 'video' ? 'var(--accent)' : dsType === 'image' ? 'var(--info)' : 'var(--warning)'}
						<div class="flex items-center gap-2 px-2 py-1.5 text-[11px]" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
							<span class="w-1.5 h-1.5 rounded-full flex-shrink-0" style="background: {dsColor};"></span>
							<span class="font-medium" style="color: {dsColor};">{dsType}</span>
							{#if dsType !== 'audio'}
								<span class="tabular-nums" style="color: var(--text-primary);">{ds.resolution_w || 768}x{ds.resolution_h || 512}</span>
							{/if}
							{#if dsType === 'video'}
								<span class="tabular-nums" style="color: var(--text-muted);">{ds.target_frames || 33}f</span>
							{/if}
							<span class="tabular-nums" style="color: var(--text-muted);">x{ds.num_repeats || 1}</span>
							<span class="font-mono truncate flex-1 text-right" style="color: var(--text-muted);">{ds.directory ? ds.directory.split(/[\\/]/).pop() : '...'}</span>
						</div>
					{/each}
					{#if valDatasets.length > 0}
						<div class="text-[9px] font-medium uppercase tracking-wider pt-1.5 px-1" style="color: var(--text-muted);">Validation</div>
						{#each valDatasets as ds, i}
							{@const dsType = ds.type || 'video'}
							{@const dsColor = dsType === 'video' ? 'var(--accent)' : dsType === 'image' ? 'var(--info)' : 'var(--warning)'}
							<div class="flex items-center gap-2 px-2 py-1.5 text-[11px]" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border-left: 2px solid var(--info);">
								<span class="w-1.5 h-1.5 rounded-full flex-shrink-0" style="background: {dsColor};"></span>
								<span class="font-medium" style="color: {dsColor};">{dsType}</span>
								{#if dsType !== 'audio'}
									<span class="tabular-nums" style="color: var(--text-primary);">{ds.resolution_w || 768}x{ds.resolution_h || 512}</span>
								{/if}
								{#if dsType === 'video'}
									<span class="tabular-nums" style="color: var(--text-muted);">{ds.target_frames || 33}f</span>
								{/if}
								<span class="tabular-nums" style="color: var(--text-muted);">x{ds.num_repeats || 1}</span>
								<span class="font-mono truncate flex-1 text-right" style="color: var(--text-muted);">{ds.directory ? ds.directory.split(/[\\/]/).pop() : '...'}</span>
							</div>
						{/each}
					{/if}
				</div>
			{/if}
		</div>

		<!-- Diagnostics -->
		{#if diagnostics.total > 0}
			<div class="p-3" style="background: var(--bg-surface); border: 1px solid {diagnostics.errors.length > 0 ? 'var(--danger)' : 'var(--warning)'}; border-radius: var(--radius-md);">
				<div class="text-[10px] font-medium uppercase tracking-wider mb-2" style="color: var(--text-muted);">
					{diagnostics.errors.length} error{diagnostics.errors.length !== 1 ? 's' : ''}{diagnostics.warnings.length > 0 ? `, ${diagnostics.warnings.length} warning${diagnostics.warnings.length !== 1 ? 's' : ''}` : ''}
				</div>
				<div class="space-y-1">
					{#each diagnostics.errors as d}
						<a href={d.href} class="flex items-center gap-2 px-2 py-1.5 text-[11px]" style="background: var(--danger-muted); border-radius: var(--radius-sm);"
							onmouseenter={(e) => { e.currentTarget.style.background = 'color-mix(in srgb, var(--danger) 15%, transparent)'; }}
							onmouseleave={(e) => { e.currentTarget.style.background = 'var(--danger-muted)'; }}
						>
							<svg class="w-3 h-3 flex-shrink-0" style="color: var(--danger);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M6 18L18 6M6 6l12 12"/></svg>
							<span class="font-medium" style="color: var(--danger);">{d.stage}</span>
							<span class="flex-1" style="color: var(--text-secondary);">{d.msg}</span>
						</a>
					{/each}
					{#each diagnostics.warnings as d}
						<a href={d.href} class="flex items-center gap-2 px-2 py-1.5 text-[11px]" style="background: var(--bg-elevated); border-radius: var(--radius-sm);">
							<svg class="w-3 h-3 flex-shrink-0" style="color: var(--warning);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01"/></svg>
							<span class="font-medium" style="color: var(--warning);">{d.stage}</span>
							<span class="flex-1" style="color: var(--text-secondary);">{d.msg}</span>
						</a>
					{/each}
				</div>
			</div>
		{:else}
			<div class="px-3 py-2 flex items-center gap-2" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
				<svg class="w-3.5 h-3.5 flex-shrink-0" style="color: var(--success);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M5 13l4 4L19 7"/></svg>
				<span class="text-[11px] font-medium" style="color: var(--success);">Ready to train</span>
			</div>
		{/if}
	</div>
{/if}

<style>
	.workflow-panel {
		background: color-mix(in srgb, var(--accent) 4%, var(--bg-surface));
		border: 1px solid var(--border-subtle);
		border-radius: var(--radius-md);
	}

	.workflow-main {
		flex: 1 1 auto;
	}

	.workflow-stages {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.4rem;
	}

	.workflow-stage {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		min-height: 2rem;
		padding: 0.42rem 0.7rem;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-secondary);
		font-size: 12px;
		font-weight: 600;
		transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
	}

	.workflow-stage:hover {
		background: color-mix(in srgb, var(--accent) 8%, var(--bg-elevated));
		border-color: color-mix(in srgb, var(--accent) 28%, var(--border));
		color: var(--text-primary);
	}

	.workflow-stage-active {
		background: color-mix(in srgb, var(--accent) 12%, var(--bg-elevated));
		border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
		box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 16%, transparent);
		color: var(--text-primary);
	}

	.workflow-stage-complete {
		background: color-mix(in srgb, var(--accent) 6%, var(--bg-elevated));
		color: var(--text-primary);
	}

	.workflow-stage-number {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1rem;
		height: 1rem;
		border-radius: var(--radius-full);
		background: color-mix(in srgb, var(--text-muted) 16%, transparent);
		color: var(--text-primary);
		font-size: 10px;
		font-variant-numeric: tabular-nums;
	}

	.workflow-stage-active .workflow-stage-number {
		background: var(--accent);
		color: var(--bg-base);
	}

	.workflow-stage-arrow {
		color: var(--text-muted);
		font-size: 12px;
	}

	.workflow-progress-zone {
		display: flex;
		flex-direction: column;
		justify-content: center;
		height: 4.8rem;
		margin-top: 0.7rem;
		overflow: hidden;
	}

	.workflow-progress-zone-visible {
		height: 4.8rem;
	}

	.workflow-idle-summary {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		min-width: 0;
	}

	.workflow-progress-copy {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 0.4rem;
	}

	.workflow-progress-text {
		display: flex;
		flex-wrap: nowrap;
		align-items: baseline;
		gap: 0.35rem 0.65rem;
		min-width: 0;
		overflow: hidden;
	}

	.workflow-idle-copy {
		display: flex;
		flex-wrap: nowrap;
		align-items: baseline;
		gap: 0.65rem;
		line-height: 1.35;
		min-width: 0;
		overflow: hidden;
	}

	.workflow-idle-metrics {
		display: flex;
		flex-wrap: nowrap;
		gap: 0.45rem;
		overflow-x: auto;
		overflow-y: hidden;
		padding-bottom: 1px;
		scrollbar-width: none;
	}

	.workflow-idle-metrics::-webkit-scrollbar {
		display: none;
	}

	.workflow-idle-metric {
		display: inline-flex;
		flex: 0 0 auto;
		align-items: baseline;
		gap: 0.4rem;
		min-height: 2rem;
		padding: 0.35rem 0.6rem;
		background: color-mix(in srgb, var(--bg-elevated) 72%, transparent);
		border: 1px solid color-mix(in srgb, var(--border) 85%, transparent);
		border-radius: var(--radius-sm);
	}

	.workflow-idle-metric-value {
		color: var(--text-primary);
		font-size: 13px;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		line-height: 1;
	}

	.workflow-idle-metric-label {
		color: var(--text-muted);
		font-size: 10px;
		font-weight: 600;
		line-height: 1;
		text-transform: uppercase;
	}

	.workflow-idle-spinner {
		width: 0.85rem;
		height: 0.85rem;
		border: 2px solid color-mix(in srgb, var(--text-muted) 30%, transparent);
		border-top-color: var(--accent);
		border-radius: var(--radius-full);
		animation: workflow-spin 700ms linear infinite;
		transform: translateY(0.08rem);
	}

	@keyframes workflow-spin {
		to {
			transform: translateY(0.08rem) rotate(360deg);
		}
	}

	.workflow-progress {
		position: relative;
		height: 0.65rem;
		overflow: hidden;
		background: color-mix(in srgb, var(--bg-elevated) 78%, var(--bg-base));
		border: 1px solid var(--border);
		border-radius: var(--radius-full);
	}

	.workflow-progress-zone-visible .workflow-progress {
		border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
		box-shadow:
			0 0 0 1px color-mix(in srgb, var(--accent) 10%, transparent),
			0 0 14px color-mix(in srgb, var(--accent) 16%, transparent),
			inset 0 0 6px rgba(0, 0, 0, 0.22);
	}

	.workflow-progress-fill {
		position: relative;
		height: 100%;
		min-width: 0;
		background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 70%, var(--info)), var(--accent));
		border-radius: inherit;
		box-shadow:
			0 0 10px color-mix(in srgb, var(--accent) 34%, transparent),
			0 0 20px color-mix(in srgb, var(--accent) 18%, transparent);
		transition: width 180ms ease;
	}

	.workflow-progress-fill::after {
		content: '';
		position: absolute;
		inset: 0;
		background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.18), transparent);
		animation: workflow-progress-glow 1800ms ease-in-out infinite;
		transform: translateX(-100%);
	}

	@keyframes workflow-progress-glow {
		0% {
			transform: translateX(-100%);
			opacity: 0;
		}
		45% {
			opacity: 1;
		}
		100% {
			transform: translateX(100%);
			opacity: 0;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.workflow-progress-fill::after {
			animation: none;
			opacity: 0;
		}
	}

	.workflow-progress-label {
		color: var(--text-primary);
		font-size: 13px;
		font-weight: 650;
		line-height: 1.35;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.workflow-progress-detail {
		color: var(--text-muted);
		font-size: 11px;
		font-variant-numeric: tabular-nums;
		line-height: 1.35;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.workflow-progress-percent {
		color: var(--text-primary);
		font-size: 15px;
		font-weight: 750;
		font-variant-numeric: tabular-nums;
		line-height: 1;
	}

	.workflow-actions {
		display: flex;
		flex: 0 0 auto;
		flex-wrap: wrap;
		gap: 0.5rem;
		justify-content: flex-end;
	}

	.workflow-primary-action,
	.workflow-secondary-action,
	.workflow-stop-action {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-height: 2.4rem;
		padding: 0.58rem 0.95rem;
		border-radius: var(--radius-sm);
		font-family: var(--font-label);
		font-size: 12px;
		font-weight: 700;
		white-space: nowrap;
		transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
	}

	.workflow-primary-action {
		min-width: 13rem;
		background: var(--accent);
		border: 1px solid color-mix(in srgb, var(--accent) 80%, var(--border));
		box-shadow: var(--shadow-sm), var(--glow-accent);
		color: var(--bg-base);
	}

	.workflow-secondary-action {
		background: color-mix(in srgb, var(--accent) 8%, var(--bg-elevated));
		border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--border));
		color: var(--text-primary);
	}

	.workflow-secondary-action:hover {
		background: color-mix(in srgb, var(--accent) 13%, var(--bg-elevated));
		border-color: color-mix(in srgb, var(--accent) 36%, var(--border));
	}

	.workflow-stop-action {
		background: var(--bg-elevated);
		border: 1px solid color-mix(in srgb, var(--danger) 34%, var(--border));
		color: var(--danger);
	}

	.workflow-stop-action:hover:not(:disabled) {
		background: var(--danger-muted);
	}

	@media (max-width: 640px) {
		.workflow-actions,
		.workflow-primary-action,
		.workflow-secondary-action,
		.workflow-stop-action {
			width: 100%;
		}

		.workflow-actions {
			justify-content: stretch;
		}
	}

	.choice-pill {
		padding: 0.45rem 0.7rem;
		font-size: 11px;
		font-weight: 500;
		border-radius: var(--radius-sm);
		transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
	}

	.choice-pill:hover {
		border-color: color-mix(in srgb, var(--accent) 22%, var(--border));
		color: var(--text-primary);
	}

	.choice-pill:disabled {
		cursor: not-allowed;
		opacity: 0.55;
	}

	.choice-pill:disabled:hover {
		color: var(--text-secondary);
	}

	.project-name-field {
		position: relative;
	}

	.project-name-status {
		position: absolute;
		top: 1px;
		right: 0;
		max-width: 48%;
		overflow: hidden;
		font-size: 10px;
		font-weight: 600;
		line-height: 1.2;
		pointer-events: none;
		text-align: right;
		white-space: nowrap;
	}

	.project-name-status span {
		display: block;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.project-path-preview {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 0.55rem;
		align-items: baseline;
		font-size: 11px;
		line-height: 1.35;
	}

	.project-path-preview span {
		color: var(--text-muted);
		font-family: var(--font-label);
		font-size: 10px;
		font-weight: 600;
		text-transform: uppercase;
	}

	.project-path-preview strong {
		min-width: 0;
		overflow: hidden;
		color: var(--text-secondary);
		font-family: var(--font-mono, monospace);
		font-weight: 500;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.recent-project-list {
		overflow-y: auto;
		padding-right: 0.2rem;
	}

	.recent-project-row {
		display: flex;
		align-items: center;
		gap: 0.85rem;
		min-height: 78px;
		padding: 0.7rem 0.75rem;
		border-radius: var(--radius-sm);
	}

	.recent-project-row:hover {
		background: color-mix(in srgb, var(--bg-elevated) 72%, var(--bg-base));
	}

	.recent-project-meta {
		display: flex;
		flex-wrap: wrap;
		gap: 0.2rem 0.7rem;
		font-size: 11px;
		line-height: 1.45;
	}

	.summary-dataset {
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}

	.summary-mode {
		color: var(--accent);
		font-weight: 650;
		text-transform: uppercase;
	}

	.summary-target {
		color: var(--text-secondary);
		font-weight: 600;
	}

	.summary-kind {
		color: var(--warning);
		font-weight: 650;
	}

	.summary-output {
		color: var(--text-primary);
		font-weight: 650;
	}

	.summary-steps {
		color: var(--success);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}

	.summary-version {
		color: var(--info);
		font-weight: 600;
	}

	.summary-rank {
		color: var(--text-secondary);
		font-family: var(--font-mono, monospace);
	}

	.recent-project-actions {
		display: flex;
		flex-shrink: 0;
		gap: 0.4rem;
	}

	@media (max-width: 640px) {
		.recent-project-row {
			align-items: stretch;
			flex-direction: column;
		}

		.recent-project-actions {
			justify-content: flex-end;
		}
	}

	.removed-project-row {
		background: color-mix(in srgb, var(--bg-elevated) 72%, var(--bg-base));
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}

	.recent-project-action {
		border-radius: var(--radius-sm);
		transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
	}

	.recent-project-action-primary {
		background: color-mix(in srgb, var(--accent) 10%, var(--bg-elevated));
		color: var(--text-primary);
		border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--border));
	}

	.recent-project-action-primary:hover {
		background: color-mix(in srgb, var(--accent) 18%, var(--bg-elevated));
		border-color: color-mix(in srgb, var(--accent) 38%, var(--border));
	}

	.recent-project-action-muted {
		background: var(--bg-surface);
		color: var(--text-muted);
		border: 1px solid var(--border);
	}

	.recent-project-action-muted:hover {
		color: var(--text-primary);
		border-color: color-mix(in srgb, var(--accent) 24%, var(--border));
		background: color-mix(in srgb, var(--bg-elevated) 88%, var(--bg-base));
	}

	.title-docs-link:hover {
		background: color-mix(in srgb, var(--accent) 12%, var(--bg-elevated)) !important;
		border-color: color-mix(in srgb, var(--accent) 44%, var(--border)) !important;
	}
</style>
