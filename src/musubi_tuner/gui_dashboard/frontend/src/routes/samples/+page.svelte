<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';
	import { onMount } from 'svelte';

	let filePath = $state('');
	let saving = $state(false);
	let loading = $state(false);
	let error = $state('');
	let success = $state('');
	let fileExists = $state(false);
	let showOverwriteConfirm = $state(false);

	// List of sample prompt entries
	let entries = $state([createEntry()]);

	function createEntry() {
		return {
			prompt: '',
			w: 768,
			h: 512,
			f: 49,
			d: 42,
			s: 20,
			g: 1.0,
			l: '',
			fs: 5.0,
			n: '',
			i: ''
		};
	}

	function addEntry() {
		entries = [...entries, createEntry()];
	}

	function removeEntry(index) {
		if (entries.length <= 1) return;
		entries = entries.filter((_, i) => i !== index);
	}

	function duplicateEntry(index) {
		const clone = { ...entries[index] };
		entries = [...entries.slice(0, index + 1), clone, ...entries.slice(index + 1)];
	}

	// Convert entry to line format: prompt text --flag value --flag value
	function entryToLine(e) {
		if (!e.prompt) return '';
		let line = e.prompt;
		if (e.w && e.w !== 768) line += ` --w ${e.w}`;
		if (e.h && e.h !== 512) line += ` --h ${e.h}`;
		if (e.f && e.f !== 45) line += ` --f ${e.f}`;
		if (e.d != null && e.d !== '') line += ` --d ${e.d}`;
		if (e.s && e.s !== 20) line += ` --s ${e.s}`;
		if (e.g != null && e.g !== 1.0) line += ` --g ${e.g}`;
		if (e.l != null && e.l !== '' && e.l !== 0) line += ` --l ${e.l}`;
		if (e.fs != null && e.fs !== 5.0) line += ` --fs ${e.fs}`;
		if (e.n) line += ` --n "${e.n}"`;
		if (e.i) line += ` --i ${e.i}`;
		return line;
	}

	// Parse line back to entry
	function lineToEntry(line) {
		const e = createEntry();
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith('#')) return null;

		const parts = trimmed.split(' --');
		e.prompt = parts[0];
		for (let j = 1; j < parts.length; j++) {
			const m = parts[j].match(/^(\w+)\s+(.*)/);
			if (!m) continue;
			const key = m[1];
			const val = m[2].replace(/^"(.*)"$/, '$1');
			if (key === 'w') e.w = parseInt(val) || 768;
			else if (key === 'h') e.h = parseInt(val) || 512;
			else if (key === 'f') e.f = parseInt(val) || 49;
			else if (key === 'd') e.d = parseInt(val);
			else if (key === 's') e.s = parseInt(val) || 20;
			else if (key === 'g') e.g = parseFloat(val) || 1.0;
			else if (key === 'l') e.l = parseFloat(val) || '';
			else if (key === 'fs') e.fs = parseFloat(val) || 5.0;
			else if (key === 'n') e.n = val;
			else if (key === 'i') e.i = val;
		}
		return e;
	}

	// Generate file content from entries
	let fileContent = $derived.by(() => {
		return entries
			.map(e => entryToLine(e))
			.filter(Boolean)
			.join('\n');
	});

	let promptCount = $derived(entries.filter(e => e.prompt).length);

	onMount(() => {
		const cfg = $projectConfig;
		if (cfg) {
			const sp = cfg.training?.sample_prompts;
			filePath = sp || (cfg.project_dir ? cfg.project_dir.replace(/[\\/]$/, '') + '/sample_prompts.txt' : 'sample_prompts.txt');
			loadFile();
		}
	});

	async function loadFile() {
		if (!filePath) return;
		error = '';
		success = '';
		loading = true;
		try {
			const res = await fetch(`/api/fs/read-file?path=${encodeURIComponent(filePath)}`);
			if (res.ok) {
				fileExists = true;
				const data = await res.json();
				const content = data.content || '';
				if (content.trim()) {
					const lines = content.split('\n');
					const parsed = lines.map(l => lineToEntry(l)).filter(Boolean);
					if (parsed.length > 0) {
						entries = parsed;
					}
				}
			} else {
				fileExists = false;
			}
		} catch (e) {
			// Silently ignore — file may not exist
		}
		loading = false;
	}

	async function saveFile() {
		if (!filePath) {
			error = 'Please set a file path first';
			return;
		}
		if (!filePath.endsWith('.txt')) {
			error = 'File path must end with .txt';
			return;
		}
		// Check if file exists — show inline overwrite confirmation
		try {
			const checkRes = await fetch(`/api/fs/exists?path=${encodeURIComponent(filePath)}`);
			if (checkRes.ok) {
				const info = await checkRes.json();
				if (info.exists) {
					showOverwriteConfirm = true;
					return;
				}
			}
		} catch {}
		await doSave();
	}

	async function doSave() {
		showOverwriteConfirm = false;
		saving = true;
		error = '';
		success = '';
		try {
			const res = await fetch('/api/fs/write-file', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ path: filePath, content: fileContent })
			});
			if (res.ok) {
				fileExists = true;
				success = 'Saved';
				setTimeout(() => success = '', 2000);
				projectConfig.update((c) => {
					if (!c) return c;
					return { ...c, training: { ...(c.training || {}), sample_prompts: filePath } };
				});
				saveProjectDebounced();
			} else {
				const err = await res.json();
				error = err.detail || 'Failed to save';
			}
		} catch (e) {
			error = e.message;
		}
		saving = false;
	}
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-4">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Sample Prompts</h2>
			<p class="text-[12px] mt-0.5" style="color: var(--text-muted);">Prompts for periodic sample generation during training.</p>
		</div>

		<!-- File path -->
		<div class="flex items-end gap-2">
			<div class="flex-1">
				<PathInput label="Prompts File" bind:value={filePath} showFiles tooltip="Path to sample prompts text file (.txt)" />
			</div>
			<button
				onclick={loadFile}
				disabled={loading}
				class="px-3 py-[7px] text-[12px] font-medium flex-shrink-0"
				style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
				onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
				onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-secondary)'; }}
			>{loading ? 'Loading...' : 'Reload'}</button>
		</div>

		<!-- Entries -->
		{#if loading}
			<div class="space-y-3">
				{#each Array(2) as _}
					<div class="p-4 animate-pulse" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
						<div class="h-3 rounded mb-3" style="background: var(--border); width: 60%;"></div>
						<div class="h-16 rounded mb-2" style="background: var(--border);"></div>
						<div class="flex gap-2">
							<div class="h-8 rounded flex-1" style="background: var(--border);"></div>
							<div class="h-8 rounded flex-1" style="background: var(--border);"></div>
							<div class="h-8 rounded flex-1" style="background: var(--border);"></div>
						</div>
					</div>
				{/each}
			</div>
		{:else}
			<div class="space-y-3">
				{#each entries as entry, i}
					<div class="p-4 relative" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); overflow: hidden;">
						<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.3;"></div>

						<!-- Header -->
						<div class="flex items-center justify-between mb-3">
							<span class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--accent);">Prompt #{i + 1}</span>
							<div class="flex items-center gap-1">
								<button
									onclick={() => duplicateEntry(i)}
									class="w-6 h-6 flex items-center justify-center"
									style="color: var(--text-muted); border-radius: var(--radius-sm);"
									onmouseenter={(e) => { e.currentTarget.style.color = 'var(--accent)'; }}
									onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
									title="Duplicate"
								>
									<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
								</button>
								{#if entries.length > 1}
									<button
										onclick={() => removeEntry(i)}
										class="px-2 py-0.5 text-[10px] font-medium"
										style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);"
										onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
										onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
									>
										Remove
									</button>
								{/if}
							</div>
						</div>

						<!-- Prompt textarea -->
						<!-- svelte-ignore a11y_label_has_associated_control -->
						<label class="block mb-3">
							<span class="block text-[10px] font-medium mb-1" style="color: var(--text-muted);">Prompt</span>
							<textarea
								class="w-full text-[12px] px-3 py-2 resize-y"
								rows="2"
								bind:value={entry.prompt}
								placeholder="A cinematic shot of a mountain landscape at sunset, golden hour lighting"
								style="background: var(--console-bg); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none; box-shadow: inset 0 1px 3px rgba(0,0,0,.2);"
								onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
								onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
							></textarea>
						</label>

						<!-- Generation params grid -->
						<div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-2">
							<FormField label="Width" type="number" bind:value={entry.w} min={64} step={64} tooltip="Output width" />
							<FormField label="Height" type="number" bind:value={entry.h} min={64} step={64} tooltip="Output height" />
							<FormField label="Frames" type="number" bind:value={entry.f} min={1} tooltip="Number of frames" />
							<FormField label="Seed" type="number" bind:value={entry.d} tooltip="Random seed" />
							<FormField label="Steps" type="number" bind:value={entry.s} min={1} tooltip="Sampling steps" />
							<FormField label="Guidance" type="number" bind:value={entry.g} step="0.1" min={0} tooltip="Guidance scale" />
						</div>

						<div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-2">
							<FormField label="CFG Scale" type="number" bind:value={entry.l} step="0.1" placeholder="None" tooltip="CFG scale (optional)" />
							<FormField label="Flow Shift" type="number" bind:value={entry.fs} step="0.1" tooltip="Flow shift" />
							<div class="col-span-1 sm:col-span-4">
								<FormField label="Negative" bind:value={entry.n} placeholder="Optional negative prompt" tooltip="Negative prompt" />
							</div>
						</div>

						<!-- I2V conditioning -->
						<div>
							<FormField label="I2V Image (optional)" bind:value={entry.i} placeholder="path/to/first_frame.jpg" tooltip="Image-to-Video: path to conditioning image for first frame" />
						</div>
					</div>
				{/each}
			</div>

			<!-- Add entry button -->
			<button
				onclick={addEntry}
				class="w-full py-2 text-[12px] font-medium flex items-center justify-center gap-1.5"
				style="background: var(--bg-surface); border: 1px dashed var(--border); color: var(--text-muted); border-radius: var(--radius-md);"
				onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
				onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
			>
				<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 6v12m6-6H6"/></svg>
				Add Prompt
			</button>
		{/if}

		<!-- Actions + Preview -->
		<div class="flex items-center gap-3">
			<button
				onclick={saveFile}
				disabled={saving || !filePath || promptCount === 0}
				class="px-5 py-2 text-[12px] font-semibold disabled:opacity-40"
				style="background: var(--accent-muted); border: 1px solid var(--accent); color: var(--accent); border-radius: var(--radius-sm);"
				onmouseenter={(e) => { if (!e.currentTarget.disabled) e.currentTarget.style.filter = 'brightness(1.15)'; }}
				onmouseleave={(e) => { e.currentTarget.style.filter = ''; }}
			>{saving ? 'Saving...' : 'Save'}</button>
			<span class="text-[11px] tabular-nums" style="color: var(--text-muted);">{promptCount} prompt{promptCount !== 1 ? 's' : ''}</span>
			{#if success}
				<span class="text-[11px] font-medium" style="color: var(--success);">{success}</span>
			{/if}
			{#if error}
				<span class="text-[11px]" style="color: var(--danger);">{error}</span>
			{/if}
		</div>

		<!-- Overwrite confirmation -->
		{#if showOverwriteConfirm}
			<div class="px-3 py-2 flex items-center gap-2" style="background: var(--warning-muted, rgba(234,179,8,0.1)); border: 1px solid var(--warning); border-radius: var(--radius-md);">
				<svg class="w-3.5 h-3.5 flex-shrink-0" style="color: var(--warning);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
				<span class="text-[11px] flex-1" style="color: var(--warning);">File already exists. Overwrite?</span>
				<button
					onclick={() => doSave()}
					class="text-[11px] font-medium px-2 py-0.5"
					style="background: var(--warning); color: var(--bg-base); border-radius: var(--radius-sm);"
				>Overwrite</button>
				<button
					onclick={() => showOverwriteConfirm = false}
					class="text-[11px] font-medium px-2 py-0.5"
					style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm);"
				>Cancel</button>
			</div>
		{/if}

		<!-- File preview -->
		{#if fileContent}
			<FormGroup title="File Preview" collapsed={entries.length > 3}>
				<pre
					class="text-[11px] leading-5 p-3 overflow-auto max-h-48 whitespace-pre-wrap break-all font-mono mt-2"
					style="background: var(--console-bg); color: var(--console-text); border-radius: var(--radius-sm); box-shadow: inset 0 2px 6px rgba(0,0,0,.3);"
				>{fileContent}</pre>
			</FormGroup>
		{/if}
	</div>
{/if}
