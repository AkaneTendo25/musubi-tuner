<script>
	import PathInput from '$lib/components/PathInput.svelte';
	import { projectConfig, saveProjectDebounced } from '$lib/stores/project.js';
	import { onMount } from 'svelte';
	import {
		defaultModelDir,
		effectiveGemmaRoot,
		effectiveGemmaSafetensors,
		effectiveLtx2Checkpoint
	} from '$lib/utils/modelPaths.js';

	let cwd = $state('');

	onMount(async () => {
		try {
			const res = await fetch('/api/fs/cwd');
			if (res.ok) cwd = (await res.json()).cwd || '';
		} catch {}
	});

	function updateConfig(key, value) {
		projectConfig.update((config) => {
			if (!config) return config;
			return { ...config, [key]: value };
		});
		saveProjectDebounced();
	}

	let modelDir = $derived(defaultModelDir(cwd, $projectConfig));
	let defaultLtx = $derived(effectiveLtx2Checkpoint(cwd, $projectConfig, ''));
	let defaultGemmaRoot = $derived(effectiveGemmaRoot(cwd, $projectConfig, '', $projectConfig?.default_gemma_safetensors || ''));
	let defaultGemmaSafetensors = $derived(effectiveGemmaSafetensors($projectConfig, ''));
</script>

<div class="space-y-5">
	<div>
		<h2 class="text-base font-semibold" style="color: var(--text-primary);">Settings</h2>
		<p class="text-[12px]" style="color: var(--text-muted);">Global fallback paths used when caching or training fields are left blank.</p>
	</div>

	<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
		<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>
		<div class="p-5 space-y-3">
			<div>
				<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Model Directory</div>
				<div class="text-[11px]" style="color: var(--text-muted);">Default download location. If unset, the dashboard uses `{cwd || '.'}/models`.</div>
			</div>
			<PathInput
				label="model_dir"
				value={$projectConfig?.model_dir || modelDir}
				oninput={(e) => updateConfig('model_dir', e.target.value)}
				placeholder={modelDir}
				tooltip="Base directory for model downloads and fallback paths."
			/>
		</div>
	</div>

	<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
		<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>
		<div class="p-5 space-y-4">
			<div>
				<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Default Model Paths</div>
				<div class="text-[11px]" style="color: var(--text-muted);">These act as project-level fallbacks for caching and training.</div>
			</div>

			<div class="grid grid-cols-1 xl:grid-cols-2 gap-3">
				<PathInput
					label="default_ltx2_checkpoint"
					value={$projectConfig?.default_ltx2_checkpoint || defaultLtx}
					oninput={(e) => updateConfig('default_ltx2_checkpoint', e.target.value)}
					showFiles
					placeholder={defaultLtx}
					tooltip="Fallback DiT checkpoint path. Used when page-specific checkpoint fields are blank."
				/>
				<PathInput
					label="default_gemma_root"
					value={$projectConfig?.default_gemma_root || defaultGemmaRoot}
					oninput={(e) => updateConfig('default_gemma_root', e.target.value)}
					placeholder={defaultGemmaRoot}
					tooltip="Fallback Gemma directory. Used when page-specific Gemma fields are blank."
				/>
			</div>

			<PathInput
				label="default_gemma_safetensors"
				value={$projectConfig?.default_gemma_safetensors || defaultGemmaSafetensors}
				oninput={(e) => updateConfig('default_gemma_safetensors', e.target.value)}
				showFiles
				placeholder="Optional single-file Gemma weights"
				tooltip="Optional fallback single-file Gemma safetensors path."
			/>
		</div>
	</div>
</div>
