<script>
	import FormField from '$lib/components/FormField.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import StatsPanel from '$lib/components/StatsPanel.svelte';
	import VramEstimateCard from '$lib/components/VramEstimateCard.svelte';
	import { defaultModelDir, effectiveGemmaRoot, effectiveLtx2Checkpoint } from '$lib/utils/modelPaths.js';
	import { projectConfig, projectLoaded, createProject, loadProjectFromPath, saveProjectDebounced, recentProjects, removeRecentProject, restoreRecentProject, closeProject } from '$lib/stores/project.js';
	import { processStatuses } from '$lib/stores/processes.js';
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

	const LTX_DOCS_FALLBACK_URL = 'https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2/docs/ltx_2.md';
	const TEMPLATE_VARIANTS_DISABLED = true;
	const DEFAULT_SINKSGD_LEARNING_RATE = '0.001';
	const DEFAULT_SINKSGD_DORA_OFT_LEARNING_RATE = '0.0005';
	const DEFAULT_SINKSGD_OPTIMIZER_ARGS = 'spectral_normalization=True scale_lr_with_effective_batch=True normed_momentum=True momentum=0.995 nesterov=True nesterov_coef=0.8 orthogonal_sinkhorn=True sinkhorn_iterations=3';

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

	$effect(() => {
		if ($projectLoaded) return;
		for (const project of $recentProjects) {
			const path = project?.path;
			if (!path || recentProjectSummaries[path] || recentProjectSummaryLoading[path]) continue;
			void loadRecentProjectSummary(path);
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
				network_alpha: null,
				optimizer_type: 'SinkSGD_adv',
				optimizer_args: DEFAULT_SINKSGD_OPTIMIZER_ARGS,
				sinksgd_orthogonal_sinkhorn: true,
				learning_rate: null,
				lr_scheduler: 'constant',
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

	// VRAM estimation helpers
	function activeGemmaSafetensors(cfg, sectionCfg) {
		return sectionCfg?.gemma_safetensors || cfg?.default_gemma_safetensors || '';
	}

	function isNativeFp8GemmaSafetensors(path) {
		const normalized = String(path || '').toLowerCase();
		return !!normalized && (
			normalized.includes('fp8') ||
			normalized.includes('float8') ||
			normalized.includes('e4m3') ||
			normalized.includes('e5m2') ||
			normalized.includes('f8_')
		);
	}

	function gemmaEstimate(cfg, sectionCfg) {
		// Gemma 3 12B: ~24GB fp16/bf16, ~12GB native FP8 or 8-bit, ~6GB 4-bit.
		const safetensorsPath = activeGemmaSafetensors(cfg, sectionCfg);
		if (sectionCfg?.gemma_load_in_4bit) return { size: 6, label: 'Gemma 4-bit' };
		if (sectionCfg?.gemma_load_in_8bit) return { size: 12, label: 'Gemma 8-bit' };
		if (isNativeFp8GemmaSafetensors(safetensorsPath)) return { size: 12, label: 'Gemma FP8' };
		return { size: 24, label: 'Gemma' };
	}

	function vaeSize(cfg) {
		// LTX-2 VAE: ~1.5GB bf16/fp16, ~3GB fp32
		const dtype = cfg?.vae_dtype || 'bfloat16';
		return dtype === 'float32' ? 3.0 : 1.5;
	}

	function normalizeOptimizerType(value) {
		return String(value || 'SinkSGD_adv').replace(/[-_]/g, '').toLowerCase();
	}

	function optimizerBytesPerParam(value, rank) {
		const optType = normalizeOptimizerType(value);
		if (optType === 'came8bit') {
			// CAME keeps factored second/residual moments and an 8-bit first moment.
			// Rank-4 LoRA tensors sit near the quantization threshold, so keep that
			// estimate conservative.
			if (rank <= 4) return 6;
			if (rank <= 8) return 3;
			return 2;
		}
		if (optType === 'sinksgd' || optType === 'sinksgdadv') return 4;
		if (optType.includes('4bit')) return 5;
		if (optType.includes('fp8')) return 6;
		if (optType.startsWith('optimi') || optType.startsWith('torchoptimi')) return 10;
		if (optType.includes('8bit')) return 6;
		if (optType.includes('schedulefree') || optType === 'automagic') return 14;
		return 12;
	}

	function coerceBool(value) {
		if (typeof value === 'boolean') return value;
		if (value == null) return false;
		if (typeof value === 'number') return value !== 0;
		return ['1', 'true', 'yes', 'y', 'on'].includes(String(value).trim().replace(/^['"]|['"]$/g, '').toLowerCase());
	}

	function networkArgValue(rawArgs, key) {
		if (!rawArgs) return null;
		const parts = Array.isArray(rawArgs) ? rawArgs.map(String) : String(rawArgs).match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
		for (let part of parts) {
			part = part.trim().replace(/^['"]|['"]$/g, '');
			const eq = part.indexOf('=');
			if (eq < 0) continue;
			if (part.slice(0, eq).trim().toLowerCase() === key.toLowerCase()) return part.slice(eq + 1).trim();
		}
		return null;
	}

	function trainingUsesDora(t) {
		// Explicit network_args match command_builder behavior and override the GUI toggle.
		const explicitDoraOft = networkArgValue(t?.network_args, 'use_dora_oft');
		if (explicitDoraOft != null) return coerceBool(explicitDoraOft);
		const explicit = networkArgValue(t?.network_args, 'use_dora');
		if (explicit != null) return coerceBool(explicit);
		return coerceBool(t?.use_dora) || coerceBool(t?.use_dokr) || coerceBool(t?.use_dora_oft);
	}

	function trainingUsesLokr(t) {
		const networkModule = String(t?.network_module || '').trim().toLowerCase();
		return networkModule === 'networks.lokr' || coerceBool(t?.use_lokr) || coerceBool(t?.use_dokr);
	}

	function lokrFactorization(dimension, factor = -1) {
		if (factor > 0 && dimension % factor === 0) {
			let m = factor;
			let n = dimension / factor;
			if (m > n) [n, m] = [m, n];
			return [m, n];
		}
		if (factor < 0) factor = dimension;
		let m = 1;
		let n = dimension;
		const length = m + n;
		while (m < n) {
			let newM = m + 1;
			while (dimension % newM !== 0) newM += 1;
			const newN = dimension / newM;
			if (newM + newN > length || newM > factor) break;
			m = newM;
			n = newN;
		}
		if (m > n) [n, m] = [m, n];
		return [m, n];
	}

	function lokrFactor(t) {
		let rawFactor = t?.lokr_factor;
		if (rawFactor == null || rawFactor === '') rawFactor = networkArgValue(t?.network_args, 'factor');
		const factor = Number.parseInt(rawFactor ?? '-1', 10);
		return Number.isFinite(factor) && factor > 0 ? factor : -1;
	}

	function lokrModuleParamCount(inDim, outDim, rank, factor) {
		const [inM, inN] = lokrFactorization(inDim, factor);
		const [outL, outK] = lokrFactorization(outDim, factor);
		const w1Params = outL * inM;
		const w2Params = rank < Math.max(outK, inN) / 2
			? rank * (outK + inN)
			: outK * inN;
		return w1Params + w2Params;
	}

	function ltx2LokrTargetShapes(mode, preset) {
		const modeValue = String(mode || 'video').trim().toLowerCase();
		const presetValue = String(preset || 't2v').trim().toLowerCase();
		const hasVideo = modeValue !== 'audio';
		const hasAudio = modeValue === 'audio' || modeValue === 'av';
		const blockCount = 48;
		const shapes = [];

		const add = (inDim, outDim, count) => {
			if (count > 0) shapes.push([inDim, outDim, count]);
		};
		const addVideoSelfAttn = () => { if (hasVideo) add(4096, 4096, 4 * blockCount); };
		const addVideoCrossAttn = () => { if (hasVideo) add(4096, 4096, 4 * blockCount); };
		const addAudioAttn = () => { if (hasAudio) add(2048, 2048, 8 * blockCount); };
		const addCrossModalAttn = () => {
			if (!(hasVideo && hasAudio)) return;
			add(4096, 2048, blockCount);
			add(2048, 2048, 2 * blockCount);
			add(2048, 4096, blockCount);
			add(2048, 2048, 2 * blockCount);
			add(4096, 2048, 2 * blockCount);
		};
		const addAllAttn = () => {
			addVideoSelfAttn();
			addVideoCrossAttn();
			addAudioAttn();
			addCrossModalAttn();
		};
		const addVideoFf = (count = blockCount) => {
			if (!hasVideo) return;
			add(4096, 16384, count);
			add(16384, 4096, count);
		};
		const addAudioFf = () => {
			if (!hasAudio) return;
			add(2048, 8192, blockCount);
			add(8192, 2048, blockCount);
		};

		if (presetValue === 't2v' || presetValue === 'lycoris') {
			addAllAttn();
		} else if (['v2v', 'av_ic', 'video_ref_only_av', 'full'].includes(presetValue)) {
			addAllAttn();
			addVideoFf();
			addAudioFf();
		} else if (presetValue === 'video_sa') {
			addVideoSelfAttn();
		} else if (presetValue === 'video_sa_ff') {
			addVideoSelfAttn();
			addVideoFf();
		} else if (presetValue === 'video_sa_ca_ff') {
			addVideoSelfAttn();
			addVideoCrossAttn();
			addVideoFf();
		} else if (presetValue === 'character_training') {
			addVideoFf(21);
		} else if (presetValue === 'audio') {
			addAudioAttn();
			addAudioFf();
		} else if (presetValue === 'audio_v2a') {
			addAudioAttn();
			addAudioFf();
			if (hasVideo && hasAudio) {
				add(2048, 2048, 2 * blockCount);
				add(4096, 2048, 2 * blockCount);
			}
		} else if (presetValue === 'audio_ref_ic' || presetValue === 'audio_ref_only_ic') {
			addAudioAttn();
			addAudioFf();
			addCrossModalAttn();
		} else {
			addAllAttn();
		}

		return shapes;
	}

	function estimateLokrTrainableParamCount({ rank, factor, mode, preset, includeDora = false }) {
		let total = 0;
		for (const [inDim, outDim, count] of ltx2LokrTargetShapes(mode, preset)) {
			total += lokrModuleParamCount(inDim, outDim, rank, factor) * count;
			if (includeDora) total += outDim * count;
		}
		return total;
	}

	function doraExtraParamRatio(rank, preset) {
		if (rank <= 0) return 0;
		const ffnHeavyPresets = new Set(['v2v', 'video_sa_ff', 'video_sa_ca_ff', 'character_training', 'audio', 'audio_ref_ic', 'audio_ref_only_ic', 'av_ic', 'video_ref_only_av', 'full']);
		const outputShare = ffnHeavyPresets.has(String(preset || 't2v').trim().toLowerCase()) ? 0.65 : 0.50;
		return Math.min(outputShare / rank, 0.25);
	}

	function doraRuntimeOverheadGB({ activationsGB, rank, preset, checkpointed }) {
		if (activationsGB <= 0) return 0;
		const ffnHeavyPresets = new Set(['v2v', 'video_sa_ff', 'video_sa_ca_ff', 'character_training', 'audio', 'audio_ref_ic', 'audio_ref_only_ic', 'av_ic', 'video_ref_only_av', 'full']);
		let baseRatio = ffnHeavyPresets.has(String(preset || 't2v').trim().toLowerCase()) ? 0.06 : 0.035;
		if (!checkpointed) baseRatio *= 1.5;
		const rankFactor = 1 + Math.max(0, 8 - rank) / 16;
		const estimate = activationsGB * baseRatio * Math.min(rankFactor, 1.4);
		return Math.min(Math.max(estimate, 0.05), 0.45);
	}

	function chunkReduction(seqLen, chunkSize) {
		if (seqLen <= 0 || chunkSize <= 0 || chunkSize >= seqLen) return 0;
		return 1 - (chunkSize / seqLen);
	}

	function datasetFrameCount(ds, fallback = 33) {
		const datasetType = String(ds?.type || 'video').trim().toLowerCase();
		if (datasetType === 'image') return 1;
		return Math.max(Number(ds?.target_frames || fallback), 1);
	}

	function datasetVramWorkload(ds) {
		const resW = Math.max(Number(ds?.resolution_w || 768), 64);
		const resH = Math.max(Number(ds?.resolution_h || 512), 64);
		const frames = datasetFrameCount(ds);
		const batchSize = Math.max(Number(ds?.batch_size || 1), 1);
		const latentFrames = Math.max(1, Math.floor((frames - 1) / 8) + 1);
		const latentHeight = Math.max(1, Math.floor(resH / 32));
		const latentWidth = Math.max(1, Math.floor(resW / 32));
		return latentFrames * latentHeight * latentWidth * batchSize;
	}

	function maxVramDataset(datasets) {
		if (!datasets?.length) return {};
		const visual = datasets.filter((d) => ['video', 'image'].includes(String(d?.type || '').trim().toLowerCase()));
		const candidates = visual.length ? visual : datasets;
		return candidates.reduce((best, current) => datasetVramWorkload(current) > datasetVramWorkload(best) ? current : best, candidates[0]);
	}

	function ffnChunkActivationMultiplier({ videoSeqLen, audioSeqLen, chunkSize, target, mode, checkpointed }) {
		const targetValue = String(target || '').trim().toLowerCase();
		if (!targetValue || targetValue === 'none') return 1;

		const chunk = Math.max(Number(chunkSize || 0), 0);
		if (chunk <= 0) return 1;

		const modeValue = String(mode || 'video').trim().toLowerCase();
		let weightedReduction = 0;
		if (modeValue === 'av') {
			const videoWeight = targetValue === 'all' || targetValue === 'video' ? 0.8 : 0;
			const audioWeight = targetValue === 'all' || targetValue === 'audio' ? 0.2 : 0;
			weightedReduction = videoWeight * chunkReduction(videoSeqLen, chunk) + audioWeight * chunkReduction(audioSeqLen, chunk);
		} else if (modeValue === 'audio') {
			weightedReduction = targetValue === 'all' || targetValue === 'audio' ? chunkReduction(audioSeqLen, chunk) : 0;
		} else {
			weightedReduction = targetValue === 'all' || targetValue === 'video' ? chunkReduction(videoSeqLen, chunk) : 0;
		}

		if (weightedReduction <= 0) return 1;
		const ffnPeakShare = checkpointed ? 0.30 : 0.12;
		const savings = Math.min(ffnPeakShare * weightedReduction, 0.35);
		return Math.max(1 - savings, 0.65);
	}

	// Three separate VRAM estimations
	function estimateLatentCaching(cfg) {
		if (!cfg?.caching) return null;
		const c = cfg.caching;
		const vae = vaeSize(c);
		// VAE activation memory scales with input resolution and frames.
		// Base: ~2.5 GB for 512x768x33f. Scales roughly proportional to pixel*frame count.
		const allDatasets = cfg?.dataset?.datasets || [];
		const ds = maxVramDataset(allDatasets);
		const resW = Math.max(Number(ds.resolution_w || 768), 64);
		const resH = Math.max(Number(ds.resolution_h || 512), 64);
		const frames = datasetFrameCount(ds);
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
		const gemma = gemmaEstimate(cfg, c);
		// Buffer for embeddings and intermediate tensors
		const buffer = 2.0;
		const total = gemma.size + buffer;
		return {
			total: Math.max(total, 1),
			parts: [
				{ label: gemma.label, value: gemma.size, color: 'var(--accent)' },
				{ label: 'Buffer', value: buffer, color: 'var(--info)' },
			]
		};
	}

	function estimateTraining(cfg) {
		if (!cfg?.training) return null;
		const t = cfg.training;
		const allDatasets = cfg?.dataset?.datasets || [];
		const ds = maxVramDataset(allDatasets);

		// ── DiT weights ──
		// LTX-2 DiT: 48 transformer blocks. VAE/Gemma NOT resident during training.
		// LTX 2.0: ~19.6B params → BF16 39 GB, FP8 19.5 GB, FP32 78 GB
		// LTX 2.3: ~21.0B params → BF16 42 GB, FP8 21 GB, FP32 84 GB
		const ltxVersion = String(t.ltx_version || '2.3');
		// Keep model weights in GiB to match byte-derived activation/optimizer estimates.
		const ditBF16 = ltxVersion === '2.3' ? 39.2 : 36.5;
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
		const presetMultiplier = { t2v: 1.0, v2v: 1.44, video_sa: 0.37, video_sa_ff: 0.56, video_sa_ca_ff: 0.74, character_training: 0.20, audio: 0.37, audio_v2a: 0.52, audio_ref_ic: 0.63, audio_ref_only_ic: 0.63, av_ic: 1.44, video_ref_only_av: 1.44, full: 2.1 }[t.lora_target_preset] || 1.0;
		const usesDora = trainingUsesDora(t);
		const doraExtraRatio = usesDora ? doraExtraParamRatio(rank, t.lora_target_preset) : 0;
		let loraParamsGB = rank * loraBasePerRank * presetMultiplier * (1 + doraExtraRatio);

		// ── Optimizer states ──
		// AdamW fp32: 12 bytes/param (fp32 master + momentum + variance)
		// AdamW 8-bit: 6 bytes/param (fp32 master + int8 momentum + int8 variance)
		// Prodigy/ScheduleFree: ~14 bytes/param
		let loraParamCount = loraParamsGB * (1024 ** 3) / 2;  // bf16 -> param count
		if (trainingUsesLokr(t)) {
			loraParamCount = estimateLokrTrainableParamCount({
				rank,
				factor: lokrFactor(t),
				mode,
				preset: t.lora_target_preset,
				includeDora: usesDora,
			});
			loraParamsGB = (loraParamCount * 2) / (1024 ** 3);
		}
		const optBytesPerParam = optimizerBytesPerParam(t.optimizer_type, rank);
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
		const sourceFrames = datasetFrameCount(ds);
		const batchSize = Math.max(Number(ds.batch_size || 1), 1);

		const latentFrames = Math.max(1, Math.floor((sourceFrames - 1) / 8) + 1);
		const latentHeight = Math.max(1, Math.floor(resolutionH / 32));
		const latentWidth = Math.max(1, Math.floor(resolutionW / 32));
		const videoSeqTokens = latentFrames * latentHeight * latentWidth;
		const audioSeqTokens = Math.round(sourceFrames);
		let seqTokens = videoSeqTokens;

		// Audio adds ~25 tokens per second of video (at 25fps)
		if (mode === 'audio') seqTokens = audioSeqTokens;  // audio-only

		const hiddenDim = mode === 'audio' ? 2048 : 4096;
		const bytesPerValue = isW8A8 ? 1 : 2;

		// Per-block activation: ~10 tensors of (batch, seq_len, hidden_dim) without checkpointing,
		// ~2 with gradient checkpointing (only block boundaries stored, recomputed in backward).
		// With blockwise checkpointing: ~1 (activations offloaded to CPU).
		const checkpointed = t.gradient_checkpointing !== false || !!t.blockwise_checkpointing;
		const activCoeff = (t.gradient_checkpointing === false) ? 10 : (t.blockwise_checkpointing ? 1 : 2);

		// Effective stored layers: without checkpointing all 48; with checkpointing, 48 boundaries
		// are stored but each is small (~1 tensor). Plus one block's full activations during recompute.
		const effectiveLayers = (t.gradient_checkpointing === false) ? totalBlocks : (t.blockwise_checkpointing ? 2 : totalBlocks);

		const perLayerBytes = activCoeff * batchSize * seqTokens * hiddenDim * bytesPerValue;
		let activations = (perLayerBytes * effectiveLayers) / (1024 ** 3);

		// Audio stream adds ~25% activation overhead in AV mode (hidden_dim=2048, separate path)
		if (isAV) activations *= 1.25;

		// Memory-saving techniques
		activations *= ffnChunkActivationMultiplier({
			videoSeqLen: videoSeqTokens,
			audioSeqLen: audioSeqTokens,
			chunkSize: t.ffn_chunk_size || 0,
			target: t.ffn_chunk_target,
			mode,
			checkpointed,
		});
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

		// ── DoRA runtime ──
		const doraRuntimeOverhead = usesDora ? doraRuntimeOverheadGB({
			activationsGB: activationTotal,
			rank,
			preset: t.lora_target_preset,
			checkpointed,
		}) : 0;

		const total = dit + loraParamsGB + optimStates + loraGrads + activationTotal + gradAccumOverhead + preservationOverhead + selfFlowOverhead + crepaOverhead + doraRuntimeOverhead;

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
		if (doraRuntimeOverhead > 0) parts.push({ label: 'DoRA', value: doraRuntimeOverhead, color: 'var(--secondary, var(--warning))' });
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
	let vramLatent = $derived(estimateLatentCaching(cfg));
	let vramText = $derived(estimateTextCaching(cfg));
	let vramTrain = $derived(estimateTraining(cfg));
	let vramTotal = $derived(systemInfo?.gpus?.[0] ? Math.round(systemInfo.gpus[0].vram_total_mb / 1024) : 24);
	let gpu = $derived(systemInfo?.gpus?.[0]);

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
		<!-- Header -->
		<div class="flex items-center gap-3">
			<input
				type="text"
				value={cfg?.name || ''}
				oninput={(e) => updateConfig('name', e.target.value)}
				class="text-base font-semibold bg-transparent border-none outline-none flex-1 min-w-0"
				style="color: var(--text-primary); padding: 0;"
			/>
			<span class="text-[10px] font-medium px-2 py-0.5" style="background: var(--accent-muted); color: var(--accent); border-radius: var(--radius-full);">{t.ltx2_mode || 'video'}</span>
			<span class="text-[10px] font-medium px-2 py-0.5" style="background: var(--bg-elevated); color: var(--text-muted); border-radius: var(--radius-full);">{t.fp8_base ? 'FP8' : 'BF16'}</span>
		</div>
		<div class="text-[11px] font-mono truncate -mt-3" style="color: var(--text-muted);">{cfg?.project_dir || ''}</div>

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
				<div class="text-[13px] font-semibold" style="color: var(--text-primary);">{t.learning_rate ?? (t.use_dora_oft ? DEFAULT_SINKSGD_DORA_OFT_LEARNING_RATE : DEFAULT_SINKSGD_LEARNING_RATE)}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Dim</div>
				<div class="text-[13px] font-semibold tabular-nums" style="color: var(--text-primary);">{t.network_dim || 16}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Alpha</div>
				<div class="text-[13px] font-semibold tabular-nums" style="color: var(--text-primary);">{t.network_alpha ?? (t.network_dim || 4)}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Optimizer</div>
				<div class="text-[13px] font-semibold truncate" style="color: var(--text-primary);">{t.optimizer_type || 'SinkSGD_adv'}</div>
			</div>
			<div>
				<div class="text-[9px] uppercase" style="color: var(--text-muted);">Scheduler</div>
				<div class="text-[13px] font-semibold truncate" style="color: var(--text-primary);">{(t.lr_scheduler || 'constant').replace('constant_with_warmup','const+warm')}</div>
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
