<script>
	import FormField from '$lib/components/FormField.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
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
			await createProject({ name: newProjectName, project_dir: newProjectDir });
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
		// VAE chunking/tiling saves ~50-70% of activation memory
		const tiling = (c.vae_chunk_size || c.vae_spatial_tile_size) ? -0.8 : 0;
		// Buffer for intermediate tensors
		const buffer = tiling ? 1.5 : 2.5;
		const total = vae + buffer + tiling;
		return {
			total: Math.max(total, 1),
			parts: [
				{ label: 'VAE', value: vae, color: 'var(--accent)' },
				{ label: 'Buffer', value: buffer + tiling, color: 'var(--info)' },
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
		const ds = cfg?.dataset?.datasets?.[0] || {};

		// LTX-2 19B base model: ~38GB fp16, ~19GB bf16, ~9.5GB fp8
		let ditBase = t.fp8_base ? 9.5 : (t.mixed_precision === 'fp16' ? 38 : 19);

		// Block swapping: each block ~0.4GB for bf16, ~0.2GB for fp8
		const blockSize = t.fp8_base ? 0.2 : 0.4;
		const swapSavings = (t.blocks_to_swap || 0) * blockSize;
		const dit = Math.max(ditBase - swapSavings, 0.5);

		// Gemma text encoder (NOT loaded if precached and no preservation techniques)
		const needsGemma = !t.use_precached_sample_prompts || t.blank_preservation || t.dop || t.prior_divergence;
		const gemma = needsGemma ? gemmaSize(t) : 0;

		// VAE is NEVER loaded during training (only during sampling, and only if not offloaded)
		// Training only uses pre-cached latents
		const vae = 0;

		// LoRA parameters: scales with rank and target layers
		const rank = t.network_dim || 16;
		const targetMultiplier = { t2v: 1.0, v2v: 1.4, audio: 0.8, full: 2.0 }[t.lora_target_preset] || 1.0;
		const loraParams = rank * rank * targetMultiplier * 0.000012; // ~0.2GB for rank 32 t2v

		// Optimizer states: AdamW stores 2x params (momentum + variance)
		const optimMultiplier = (t.optimizer_type || 'adamw8bit').includes('8bit') ? 1.5 : 2.0;
		const optimStates = loraParams * optimMultiplier;

		// Activations: more realistic calculation
		// Base activation memory scales with: hidden_dim × num_layers × sequence_length × batch_size
		const res = (ds.resolution_w || 768) * (ds.resolution_h || 512) / (768 * 512); // normalized to 768x512
		const frames = Math.min(ds.target_frames || 33, 129);
		const batchSize = ds.batch_size || 1;

		// Sequence length for transformer: (H/patchsize) × (W/patchsize) × frames
		// For LTX-2: patch_size=2 for video, so 768x512 → 384x256 → ~98k patches for 1 frame
		// At 33 frames: ~3.2M patches (sequence length)
		const patchesPerFrame = res * (384 * 256); // base is 384x256 patches
		const seqLen = patchesPerFrame * frames;

		// Activation memory: hidden_dim (3840 for LTXAV) × layers (48) × seq_len × batch × dtype(2 bytes)
		// But we only store activations for layers being recomputed
		// With gradient checkpointing: store ~10-15% of layers
		const hiddenDim = 3840;
		const numLayers = 48 - (t.blocks_to_swap || 0);
		let activations = (hiddenDim * numLayers * seqLen * batchSize * 2) / (1024**3); // bytes → GB

		// Gradient checkpointing: only store activations for checkpoint layers (~10%)
		if (t.gradient_checkpointing !== false) {
			activations *= 0.1;
			// Blockwise checkpointing: even more aggressive
			if (t.blockwise_checkpointing) activations *= 0.5;
			// CPU offloading: offload checkpointed activations to CPU (minimal VRAM)
			if (t.gradient_checkpointing_cpu_offload) activations *= 0.2;
		}

		// Add gradient storage for LoRA params
		const loraGrads = loraParams;

		// Gradient accumulation: we accumulate gradients before optimizer step
		// This doesn't increase activation memory, but we need to store accumulated grads
		const gradAccum = t.gradient_accumulation_steps || 1;
		const gradAccumOverhead = gradAccum > 1 ? loraGrads * 0.5 : 0;

		// Preservation techniques: each adds extra forward passes (more activations)
		let preservationOverhead = 0;
		if (t.blank_preservation) preservationOverhead += activations * 0.5; // 2 extra forwards
		if (t.dop) preservationOverhead += activations * 0.5;
		if (t.prior_divergence) preservationOverhead += activations * 0.25; // 1 extra forward

		// CREPA: projector params (~33M for backbone mode) + feature buffers
		const crepaOverhead = t.crepa ? 0.15 : 0;

		// Offloading savings
		let offloadSave = 0;
		if (t.img_in_txt_in_offloading) offloadSave += 1.2;
		if (t.sample_with_offloading) offloadSave += 0; // only affects sampling, not training

		// Total
		const total = dit + gemma + vae + loraParams + optimStates + activations + loraGrads + gradAccumOverhead + preservationOverhead + crepaOverhead - offloadSave;

		// Build parts breakdown
		const parts = [
			{ label: 'DiT', value: dit, color: 'var(--accent)' },
		];
		if (gemma > 0) parts.push({ label: 'Gemma', value: gemma, color: 'var(--info)' });
		parts.push({ label: 'LoRA', value: loraParams, color: 'var(--warning)' });
		parts.push({ label: 'Optimizer', value: optimStates, color: 'var(--warning)' });
		parts.push({ label: 'Activ.', value: activations + loraGrads, color: 'var(--success)' });
		if (gradAccumOverhead > 0) parts.push({ label: 'GradAccum', value: gradAccumOverhead, color: 'var(--info)' });
		if (preservationOverhead > 0) parts.push({ label: 'Preserv.', value: preservationOverhead, color: 'var(--danger)' });
		if (crepaOverhead > 0) parts.push({ label: 'CREPA', value: crepaOverhead, color: 'var(--secondary, var(--info))' });

		return {
			total: Math.max(total, 2),
			parts,
			swap: swapSavings,
			offloadSave,
			noGemma: !needsGemma,
			noPrecache: needsGemma,
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
		if (!c.ltx2_checkpoint && !t.ltx2_checkpoint) {
			errors.push({ stage: 'Caching', msg: 'LTX-2 checkpoint path not set', href: '/caching' });
		}
		if (!c.gemma_root && !t.gemma_root) {
			errors.push({ stage: 'Caching', msg: 'Gemma text encoder path not set', href: '/caching' });
		}

		// Training
		if (!t.ltx2_checkpoint) {
			warnings.push({ stage: 'Training', msg: 'Training checkpoint not set', href: '/training' });
		}
		if (!t.gemma_root) {
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
					<PathInput label="Project Directory" bind:value={newProjectDir} placeholder="{cwd ? cwd + '/projects/Project_Name' : 'Path to store project files'}" tooltip="Directory where project.json and configs will be saved" />
					<button
						onclick={handleCreate}
						disabled={!newProjectDir || creating}
						class="w-full py-2.5 text-[13px] font-semibold disabled:opacity-40"
						style="background: var(--accent); color: var(--bg-base); border-radius: var(--radius-sm); box-shadow: var(--shadow-sm), var(--glow-accent);"
					>{creating ? 'Creating...' : 'Create Project'}</button>
				</div>
			</div>

			<!-- Right: Recent Projects + Load Existing -->
			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Open Project</div>

				{#if $recentProjects.length > 0}
					<div class="space-y-1.5">
						<p class="text-[11px] font-medium" style="color: var(--text-muted);">Recent</p>
						{#each $recentProjects as proj}
							<div class="flex items-center gap-2 group">
								<button
									onclick={() => { loadPath = proj.path; handleLoad(); }}
									class="flex-1 flex items-center gap-3 px-3 py-2 text-left min-w-0"
									style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);"
									onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; }}
									onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border-subtle)'; }}
								>
									<div class="min-w-0 flex-1">
										<div class="text-[12px] font-semibold truncate" style="color: var(--text-primary);">{proj.name}</div>
										<div class="text-[10px] font-mono truncate" style="color: var(--text-muted);">{proj.path}</div>
									</div>
								</button>
								<button
									onclick={() => removeRecentProject(proj.path)}
									class="flex-shrink-0 px-2 py-0.5 text-[10px] font-medium opacity-0 group-hover:opacity-100 transition-opacity"
									style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);"
									onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
									onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
								>
									Remove
								</button>
							</div>
						{/each}
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
							<span class="text-[9px] mt-0.5" style="color: var(--info);">prompts precached</span>
						{/if}
					</div>
				{/if}
			</div>
			{#if vramTrain && vramTrain.total > vramTotal}
				<div class="mt-2 text-[10px] px-2 py-1 flex items-center gap-1.5" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">
					<svg class="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
					Training may exceed VRAM — try FP8, quantization, or blocks_to_swap.
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
