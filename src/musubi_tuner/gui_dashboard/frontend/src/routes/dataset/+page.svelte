<script>
	import FormToggle from '$lib/components/FormToggle.svelte';
	import DatasetEntry from '$lib/components/DatasetEntry.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced, saveProjectNow } from '$lib/stores/project.js';

	let tomlPreview = $state('');
	let showToml = $state(false);
	let saving = $state(false);

	function addDataset(isValidation = false) {
		projectConfig.update((c) => {
			if (!c) return c;
			const entry = {
				type: 'video',
				directory: '',
				cache_directory: '',
				reference_cache_directory: '',
				control_directory: '',
				jsonl_file: '',
				resolution_w: 768,
				resolution_h: 512,
				batch_size: 1,
				num_repeats: 1,
				caption_extension: '.txt',
				target_frames: 33,
				frame_extraction: 'head',
				frame_sample: null,
				max_frames: null,
				frame_stride: null,
				source_fps: null,
				target_fps: null
			};
			if (isValidation) {
				c.dataset.validation_datasets = [...(c.dataset.validation_datasets || []), entry];
			} else {
				c.dataset.datasets = [...(c.dataset.datasets || []), entry];
			}
			return c;
		});
		saveProjectDebounced();
	}

	function removeDataset(index, isValidation = false) {
		projectConfig.update((c) => {
			if (!c) return c;
			if (isValidation) {
				c.dataset.validation_datasets = c.dataset.validation_datasets.filter((_, i) => i !== index);
			} else {
				c.dataset.datasets = c.dataset.datasets.filter((_, i) => i !== index);
			}
			return c;
		});
		saveProjectDebounced();
	}

	function updateGeneral(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			c.dataset.general[key] = value;
			return c;
		});
		saveProjectDebounced();
	}

	async function exportToml() {
		saving = true;
		try {
			await saveProjectNow();
			const res = await fetch('/api/dataset/export-toml', { method: 'POST' });
			if (!res.ok) {
				const err = await res.json();
				alert(err.detail || 'Failed to export TOML');
			} else {
				const data = await res.json();
				alert(`TOML saved to: ${data.path}`);
			}
		} catch (e) {
			alert(`Error: ${e.message}`);
		}
		saving = false;
	}

	async function previewToml() {
		showToml = !showToml;
		if (showToml) {
			try {
				await saveProjectNow();
				const res = await fetch('/api/dataset/preview-toml');
				if (res.ok) {
					const data = await res.json();
					tomlPreview = data.toml;
				}
			} catch {
				tomlPreview = 'Failed to generate preview';
			}
		}
	}

	let datasets = $derived($projectConfig?.dataset?.datasets || []);
	let validationDatasets = $derived($projectConfig?.dataset?.validation_datasets || []);
	let general = $derived($projectConfig?.dataset?.general || {});
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-5">
		<div class="flex items-center justify-between">
			<div>
				<h2 class="text-base font-semibold" style="color: var(--text-primary);">Dataset</h2>
				<p class="text-[12px]" style="color: var(--text-muted);">Configure training and validation datasets.</p>
			</div>
			<div class="flex items-center gap-5">
				<FormToggle label="Enable Bucket" checked={general.enable_bucket ?? true} onchange={(e) => updateGeneral('enable_bucket', e.target.checked)} tooltip="Enable resolution bucketing for varied aspect ratios" />
				<FormToggle label="No Upscale" checked={general.bucket_no_upscale ?? true} onchange={(e) => updateGeneral('bucket_no_upscale', e.target.checked)} tooltip="Prevent upscaling images smaller than bucket resolution" />
			</div>
		</div>

		<!-- Training Datasets -->
		<div>
			<div class="flex items-center justify-between mb-2">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Training</span>
				<button
					onclick={() => addDataset(false)}
					class="px-3 py-1 text-[11px] font-medium"
					style="color: var(--accent); border: 1px solid var(--accent-muted); border-radius: var(--radius-full); background: var(--accent-subtle-bg);"
				>+ Add</button>
			</div>
			<div class="grid grid-cols-1 xl:grid-cols-2 gap-3">
				{#each datasets as entry, i}
					<DatasetEntry
						bind:entry={$projectConfig.dataset.datasets[i]}
						index={i}
						onRemove={() => removeDataset(i, false)}
					/>
				{/each}
			</div>
			{#if datasets.length === 0}
				<div class="text-center py-8 text-[12px]" style="border: 1px dashed var(--border); border-radius: var(--radius-sm); color: var(--text-muted);">
					No datasets. Click "+ Add" to start.
				</div>
			{/if}
		</div>

		<!-- Validation Datasets -->
		<div>
			<div class="flex items-center justify-between mb-2">
				<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Validation</span>
				<button
					onclick={() => addDataset(true)}
					class="px-3 py-1 text-[11px] font-medium"
					style="color: var(--accent); border: 1px solid var(--accent-muted); border-radius: var(--radius-full); background: var(--accent-subtle-bg);"
				>+ Add</button>
			</div>
			<div class="grid grid-cols-1 xl:grid-cols-2 gap-3">
				{#each validationDatasets as entry, i}
					<DatasetEntry
						bind:entry={$projectConfig.dataset.validation_datasets[i]}
						index={i}
						onRemove={() => removeDataset(i, true)}
					/>
				{/each}
			</div>
			{#if validationDatasets.length === 0}
				<div class="text-center py-3 text-[11px]" style="color: var(--text-muted);">None</div>
			{/if}
		</div>

		<!-- Bottom -->
		<div class="flex items-center gap-2 pt-3" style="border-top: 1px solid var(--border-subtle);">
			<button
				onclick={exportToml}
				disabled={saving}
				class="px-4 py-1.5 text-[12px] font-semibold disabled:opacity-40"
				style="background: var(--accent-muted); border: 1px solid var(--accent); color: var(--accent); border-radius: var(--radius-sm);"
				onmouseenter={(e) => { if (!e.currentTarget.disabled) e.currentTarget.style.filter = 'brightness(1.15)'; }}
				onmouseleave={(e) => { e.currentTarget.style.filter = ''; }}
			>{saving ? 'Saving...' : 'Export TOML'}</button>
			<button
				onclick={previewToml}
				class="px-4 py-1.5 text-[12px] font-medium"
				style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
			>{showToml ? 'Hide' : 'Preview'}</button>
		</div>

		{#if showToml}
			<pre class="p-3 text-[11px] font-mono overflow-auto max-h-80" style="background: var(--console-bg); border: 1px solid var(--border-subtle); color: var(--text-secondary); border-radius: var(--radius-sm);">{tomlPreview}</pre>
		{/if}
	</div>
{/if}
