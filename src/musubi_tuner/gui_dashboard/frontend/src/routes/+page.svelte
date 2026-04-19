<script>
	import FormField from '$lib/components/FormField.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import StatsPanel from '$lib/components/StatsPanel.svelte';
	import { projectConfig, projectLoaded, createProject, loadProjectFromPath, saveProjectDebounced, recentProjects, removeRecentProject, closeProject } from '$lib/stores/project.js';
	import { processStatuses } from '$lib/stores/processes.js';
	import { status } from '$lib/stores/status.js';
	import { onMount } from 'svelte';

	let newProjectName = $state('New Project');
	let newProjectDir = $state('');
	let loadPath = $state('');
	let error = $state('');
	let creating = $state(false);
	let systemInfo = $state(null);
	let cwd = $state('');

	// Update project directory when name changes
	$effect(() => {
		if (cwd && newProjectName) {
			const sanitized = newProjectName.replace(/[<>:"/\\|?*]/g, '_').replace(/\s+/g, '_');
			newProjectDir = `${cwd}/projects/${sanitized}`;
		}
	});

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

	async function handleCreate() {
		error = '';
		creating = true;
		try {
			await createProject({
				name: newProjectName,
				project_dir: newProjectDir,
				model_dir: cwd ? `${cwd}/models` : 'models'
			});
		} catch (e) { error = e.message; }
		creating = false;
	}

	async function handleLoad() {
		error = '';
		try { await loadProjectFromPath(loadPath); } catch (e) { error = e.message; }
	}

	function updateConfig(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, [key]: value };
		});
		saveProjectDebounced();
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
		const ltxVersion = String(t.ltx_version || '2.0');
		const ditBF16 = ltxVersion === '2.3' ? 42 : 39;
		const isFp8 = !!t.fp8_base;
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
		const presetMultiplier = { t2v: 1.0, v2v: 1.44, video_sa: 0.37, video_sa_ff: 0.56, video_sa_ca_ff: 0.74, audio: 0.52, audio_ref_only_ic: 0.63, av_ic: 1.44, video_ref_only_av: 1.44, full: 2.1 }[t.lora_target_preset] || 1.0;
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

		const hiddenDim = 4096;  // video inner_dim = 32 heads * 128 dim_head
		const bytesPerValue = isFp8 ? 1 : 2;

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
		// Shadow params (EMA teacher): clone of LoRA weights on GPU (or CPU if offloaded)
		// Projector MLP: ~0.02 GB. Extra forward activations: ~10% overhead.
		let selfFlowOverhead = 0;
		if (t.self_flow) {
			const teacherOnGPU = !t.self_flow_offload_teacher_params;
			selfFlowOverhead += teacherOnGPU ? loraParamsGB : 0;  // shadow params
			selfFlowOverhead += 0.02;  // projector MLP
			selfFlowOverhead += activationTotal * 0.10;  // extra forward activation overhead
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
	<div class="max-w-3xl mx-auto mt-10 space-y-6">
		<!-- Header -->
		<div class="text-center">
			<div class="w-12 h-12 mx-auto mb-3 flex items-center justify-center" style="background: var(--logo-bg); box-shadow: var(--logo-shadow); border-radius: var(--logo-radius); clip-path: var(--logo-clip);">
				<span class="text-base font-bold" style="color: var(--bg-base);">M</span>
			</div>
			<h2 class="text-lg font-semibold" style="color: var(--text-primary);">Musubi Tuner</h2>
			<p class="text-[12px] mt-1" style="color: var(--text-muted);">LTX-2 LoRA training management</p>
		</div>

		<!-- Two-panel layout -->
		<div class="grid grid-cols-1 md:grid-cols-2 gap-5">
			<!-- Left: Create New Project -->
			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">New Project</div>
				<div class="space-y-3">
					<FormField label="Project Name" bind:value={newProjectName} placeholder="My LTX-2 LoRA" tooltip="Display name for your project" />
					<PathInput label="Project Directory" bind:value={newProjectDir} placeholder={cwd ? `${cwd}/projects/Project_Name` : 'Path to store project files'} tooltip="Directory where project.json and configs will be saved" />
					<button
						onclick={handleCreate}
						disabled={!newProjectDir || creating}
						class="w-full py-2.5 text-[13px] font-semibold disabled:opacity-40 transition-colors"
						style="background: color-mix(in srgb, var(--accent) 78%, var(--bg-elevated)); color: var(--text-primary); border: 1px solid color-mix(in srgb, var(--accent) 42%, var(--border)); border-radius: var(--radius-sm); box-shadow: var(--shadow-sm);"
						onmouseenter={(e) => { e.currentTarget.style.background = 'color-mix(in srgb, var(--accent) 88%, var(--bg-elevated))'; e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 62%, var(--border))'; }}
						onmouseleave={(e) => { e.currentTarget.style.background = 'color-mix(in srgb, var(--accent) 78%, var(--bg-elevated))'; e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 42%, var(--border))'; }}
					>{creating ? 'Creating...' : 'Create Project'}</button>
				</div>
			</div>

			<!-- Right: Recent Projects + Load Existing -->
			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Open Project</div>

				{#if $recentProjects.length > 0}
					<div class="space-y-2.5">
						<div class="flex items-center justify-between">
							<p class="text-[11px] font-medium uppercase tracking-[0.18em]" style="color: var(--text-secondary); font-family: var(--font-label);">Recent</p>
							<span class="text-[10px] px-2 py-0.5 rounded-full" style="background: var(--bg-elevated); color: var(--text-muted); border: 1px solid var(--border);">{$recentProjects.length}</span>
						</div>
						<div class="p-2 space-y-2" style="background: color-mix(in srgb, var(--bg-elevated) 68%, transparent); border: 1px solid var(--border); border-radius: var(--radius-sm);">
						{#each $recentProjects as proj}
							<div class="flex items-center gap-2 group">
								<button
									onclick={() => { loadPath = proj.path; handleLoad(); }}
									class="flex-1 flex items-center gap-3 px-3 py-2.5 text-left min-w-0 transition-colors"
									style="background: color-mix(in srgb, var(--bg-surface) 70%, var(--bg-elevated)); border: 1px solid color-mix(in srgb, var(--accent) 20%, var(--border-subtle)); border-radius: var(--radius-sm); box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);"
									onmouseenter={(e) => { e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 55%, var(--border-subtle))'; e.currentTarget.style.background = 'color-mix(in srgb, var(--bg-surface) 82%, var(--accent-muted))'; }}
									onmouseleave={(e) => { e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 20%, var(--border-subtle))'; e.currentTarget.style.background = 'color-mix(in srgb, var(--bg-surface) 70%, var(--bg-elevated))'; }}
								>
									<div class="w-7 h-7 flex items-center justify-center flex-shrink-0 rounded-sm" style="background: color-mix(in srgb, var(--accent) 16%, var(--bg-base)); color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 22%, var(--border));">P</div>
									<div class="min-w-0 flex-1">
										<div class="text-[12px] font-semibold truncate" style="color: var(--text-primary);">{proj.name}</div>
										<div class="text-[10px] font-mono truncate mt-0.5" style="color: var(--text-secondary);">{proj.path}</div>
									</div>
								</button>
								<button
									onclick={() => removeRecentProject(proj.path)}
									class="flex-shrink-0 px-2 py-1 text-[10px] font-medium opacity-0 group-hover:opacity-100 transition-opacity"
									style="color: var(--text-muted); background: color-mix(in srgb, var(--bg-elevated) 88%, var(--bg-base)); border: 1px solid var(--border); border-radius: var(--radius-sm);"
									onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
									onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
								>
									Remove
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

		<!-- VRAM Estimation — Three Gauges -->
		<div class="p-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="flex items-center justify-between mb-3">
				<span class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">
					VRAM Estimation{#if gpu} — {gpu.name.replace('NVIDIA ','').replace('GeForce ','')}{/if}
				</span>
				<span class="text-[10px] tabular-nums" style="color: var(--text-muted);">{vramTotal} GB total</span>
			</div>
			<div class="grid grid-cols-3 gap-3">
				<!-- Latent Caching gauge -->
				{#if vramLatent}
					{@const pct = Math.min(vramLatent.total / vramTotal, 1.2)}
					{@const over = vramLatent.total > vramTotal}
					{@const col = over ? 'var(--danger)' : 'var(--accent)'}
					{@const r = 52}
					{@const circ = 2 * Math.PI * r}
					{@const off = circ * (1 - Math.min(pct, 1))}
					<div class="flex flex-col items-center">
						<div class="relative" style="width: 140px; height: 140px;">
							<svg viewBox="0 0 120 120" class="w-full h-full" style="transform: rotate(-90deg);">
								<circle cx="60" cy="60" r={r} fill="none" stroke="var(--border)" stroke-width="6" />
								<circle cx="60" cy="60" r={r} fill="none" stroke={col} stroke-width="6"
									stroke-dasharray={circ} stroke-dashoffset={off}
									stroke-linecap="round" style="transition: stroke-dashoffset 0.5s ease;" />
							</svg>
							<div class="absolute inset-0 flex flex-col items-center justify-center">
								<span class="text-[18px] font-bold tabular-nums leading-none" style="color: {col};">~{vramLatent.total.toFixed(1)}</span>
								<span class="text-[10px] mt-0.5" style="color: var(--text-muted);">/ {vramTotal} GB</span>
							</div>
						</div>
						<span class="text-[11px] font-medium mt-1.5" style="color: var(--text-secondary);">Latent Cache</span>
						<div class="flex flex-wrap justify-center gap-x-3 gap-y-0 mt-1">
							{#each vramLatent.parts as p}
								<span class="flex items-center gap-1 text-[10px]">
									<span class="w-1.5 h-1.5 rounded-full" style="background: {p.color};"></span>
									<span style="color: var(--text-muted);">{p.label} {p.value.toFixed(1)}G</span>
								</span>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Text Caching gauge -->
				{#if vramText}
					{@const pct = Math.min(vramText.total / vramTotal, 1.2)}
					{@const over = vramText.total > vramTotal}
					{@const col = over ? 'var(--danger)' : 'var(--info)'}
					{@const r = 52}
					{@const circ = 2 * Math.PI * r}
					{@const off = circ * (1 - Math.min(pct, 1))}
					<div class="flex flex-col items-center">
						<div class="relative" style="width: 140px; height: 140px;">
							<svg viewBox="0 0 120 120" class="w-full h-full" style="transform: rotate(-90deg);">
								<circle cx="60" cy="60" r={r} fill="none" stroke="var(--border)" stroke-width="6" />
								<circle cx="60" cy="60" r={r} fill="none" stroke={col} stroke-width="6"
									stroke-dasharray={circ} stroke-dashoffset={off}
									stroke-linecap="round" style="transition: stroke-dashoffset 0.5s ease;" />
							</svg>
							<div class="absolute inset-0 flex flex-col items-center justify-center">
								<span class="text-[18px] font-bold tabular-nums leading-none" style="color: {col};">~{vramText.total.toFixed(1)}</span>
								<span class="text-[10px] mt-0.5" style="color: var(--text-muted);">/ {vramTotal} GB</span>
							</div>
						</div>
						<span class="text-[11px] font-medium mt-1.5" style="color: var(--text-secondary);">Text Cache</span>
						<div class="flex flex-wrap justify-center gap-x-3 gap-y-0 mt-1">
							{#each vramText.parts as p}
								<span class="flex items-center gap-1 text-[10px]">
									<span class="w-1.5 h-1.5 rounded-full" style="background: {p.color};"></span>
									<span style="color: var(--text-muted);">{p.label} {p.value.toFixed(1)}G</span>
								</span>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Training gauge -->
				{#if vramTrain}
					{@const pct = Math.min(vramTrain.total / vramTotal, 1.2)}
					{@const over = vramTrain.total > vramTotal}
					{@const col = over ? 'var(--danger)' : 'var(--warning)'}
					{@const r = 52}
					{@const circ = 2 * Math.PI * r}
					{@const off = circ * (1 - Math.min(pct, 1))}
					<div class="flex flex-col items-center">
						<div class="relative" style="width: 140px; height: 140px;">
							<svg viewBox="0 0 120 120" class="w-full h-full" style="transform: rotate(-90deg);">
								<circle cx="60" cy="60" r={r} fill="none" stroke="var(--border)" stroke-width="6" />
								<circle cx="60" cy="60" r={r} fill="none" stroke={col} stroke-width="6"
									stroke-dasharray={circ} stroke-dashoffset={off}
									stroke-linecap="round" style="transition: stroke-dashoffset 0.5s ease;" />
							</svg>
							<div class="absolute inset-0 flex flex-col items-center justify-center">
								<span class="text-[18px] font-bold tabular-nums leading-none" style="color: {col};">~{vramTrain.total.toFixed(1)}</span>
								<span class="text-[10px] mt-0.5" style="color: var(--text-muted);">/ {vramTotal} GB</span>
							</div>
						</div>
						<span class="text-[11px] font-medium mt-1.5" style="color: var(--text-secondary);">Training</span>
						<div class="flex flex-wrap justify-center gap-x-3 gap-y-0 mt-1">
							{#each vramTrain.parts as p}
								<span class="flex items-center gap-1 text-[10px]">
									<span class="w-1.5 h-1.5 rounded-full" style="background: {p.color};"></span>
									<span style="color: var(--text-muted);">{p.label} {p.value.toFixed(1)}G</span>
								</span>
							{/each}
						</div>
						{#if vramTrain.swap > 0}
							<span class="text-[9px] mt-0.5" style="color: var(--success);">-{vramTrain.swap.toFixed(1)}G swap</span>
						{/if}
						{#if vramTrain.noGemma}
							<span class="text-[9px] mt-0.5" style="color: var(--info);">steady-state: cached latents/text (no VAE/Gemma)</span>
						{/if}
						{#if vramTrain.samplingSpike}
							<span class="text-[9px] mt-0.5" style="color: var(--text-muted);">sampling can spike VRAM (temporary VAE/Gemma load)</span>
						{/if}
						{#if vramTrain.preservationGemmaSpike}
							<span class="text-[9px] mt-0.5" style="color: var(--text-muted);">preservation prompt encoding loads Gemma once (if not precached)</span>
						{/if}
					</div>
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
