<script>
	import FormField from '$lib/components/FormField.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';

	let filePath = $state('');
	let saving = $state(false);
	let loading = $state(false);
	let error = $state('');
	let success = $state('');
	let initialized = $state(false);
	let hydratedFromProject = $state(false);
	let lastSavedSignature = $state('');
	let inlineSaveSignature = $state('');

	let entries = $state([createEntry()]);

	function createEntry() {
		return {
			prompt: '',
			w: 768,
			h: 512,
			f: 45,
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

	function updateEntry(index, key, value) {
		const next = [...entries];
		next[index] = { ...next[index], [key]: value };
		entries = next;
	}

	function entryToLine(entry) {
		if (!entry.prompt) return '';
		let line = entry.prompt;
		if (entry.w && entry.w !== 768) line += ` --w ${entry.w}`;
		if (entry.h && entry.h !== 512) line += ` --h ${entry.h}`;
		if (entry.f && entry.f !== 45) line += ` --f ${entry.f}`;
		if (entry.d != null && entry.d !== '') line += ` --d ${entry.d}`;
		if (entry.s && entry.s !== 20) line += ` --s ${entry.s}`;
		if (entry.g != null && entry.g !== 1.0) line += ` --g ${entry.g}`;
		if (entry.l != null && entry.l !== '' && entry.l !== 0) line += ` --l ${entry.l}`;
		if (entry.fs != null && entry.fs !== 5.0) line += ` --fs ${entry.fs}`;
		if (entry.n) line += ` --n "${entry.n}"`;
		if (entry.i) line += ` --i ${entry.i}`;
		return line;
	}

	function lineToEntry(line) {
		const entry = createEntry();
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith('#')) return null;

		const parts = trimmed.split(' --');
		entry.prompt = parts[0];
		for (let index = 1; index < parts.length; index++) {
			const match = parts[index].match(/^(\w+)\s+(.*)/);
			if (!match) continue;
			const key = match[1];
			const value = match[2].replace(/^"(.*)"$/, '$1');
			if (key === 'w') entry.w = parseInt(value) || 768;
			else if (key === 'h') entry.h = parseInt(value) || 512;
			else if (key === 'f') entry.f = parseInt(value) || 45;
			else if (key === 'd') entry.d = parseInt(value);
			else if (key === 's') entry.s = parseInt(value) || 20;
			else if (key === 'g') entry.g = parseFloat(value) || 1.0;
			else if (key === 'l') entry.l = parseFloat(value) || '';
			else if (key === 'fs') entry.fs = parseFloat(value) || 5.0;
			else if (key === 'n') entry.n = value;
			else if (key === 'i') entry.i = value;
		}
		return entry;
	}

	let fileContent = $derived.by(() => {
		return entries
			.map((entry) => entryToLine(entry))
			.filter(Boolean)
			.join('\n');
	});

	let promptCount = $derived(entries.filter((entry) => entry.prompt).length);
	let autosaveSignature = $derived(filePath && filePath.endsWith('.txt') ? `${filePath}\n${fileContent}` : '');

	function syncTrainingSamplePrompts(path) {
		projectConfig.update((config) => {
			if (!config) return config;
			if ((config.training?.sample_prompts || '') === path) return config;
			return {
				...config,
				training: { ...(config.training || {}), sample_prompts: path }
			};
		});
		saveProjectDebounced();
	}

	function syncTrainingSamplePromptsText(text) {
		projectConfig.update((config) => {
			if (!config) return config;
			if ((config.training?.sample_prompts_text || '') === text) return config;
			return {
				...config,
				training: { ...(config.training || {}), sample_prompts_text: text }
			};
		});
		saveProjectDebounced();
	}

	$effect(() => {
		if (hydratedFromProject || !$projectLoaded || !$projectConfig) return;
		hydratedFromProject = true;
		const config = $projectConfig;
		const samplePrompts = config.training?.sample_prompts;
		const samplePromptsText = config.training?.sample_prompts_text || '';
		filePath = samplePrompts || '';
		if (filePath) {
			void loadFile().then(() => {
				initialized = true;
			});
		} else if (samplePromptsText.trim()) {
			const parsed = samplePromptsText
				.split('\n')
				.map((line) => lineToEntry(line))
				.filter(Boolean);
			if (parsed.length > 0) {
				entries = parsed;
			}
			lastSavedSignature = `\n${samplePromptsText}`;
			inlineSaveSignature = samplePromptsText;
			initialized = true;
		} else {
			initialized = true;
		}
	});

	$effect(() => {
		if (!initialized || loading || saving || !autosaveSignature || promptCount === 0) return;
		if (autosaveSignature === lastSavedSignature) return;
		const timer = setTimeout(() => {
			void persistFile({ showMessage: false, silent: true });
		}, 900);
		return () => clearTimeout(timer);
	});

	$effect(() => {
		if (!initialized) return;
		syncTrainingSamplePrompts(filePath);
	});

	$effect(() => {
		if (!initialized) return;
		if (fileContent === inlineSaveSignature) return;
		inlineSaveSignature = fileContent;
		syncTrainingSamplePromptsText(fileContent);
	});

	async function loadFile() {
		if (!filePath) return;
		error = '';
		success = '';
		loading = true;
		try {
			const res = await fetch(`/api/fs/read-file?path=${encodeURIComponent(filePath)}`);
			if (res.ok) {
				const data = await res.json();
				const content = data.content || '';
				if (content.trim()) {
					const parsed = content
						.split('\n')
						.map((line) => lineToEntry(line))
						.filter(Boolean);
					if (parsed.length > 0) {
						entries = parsed;
					}
				}
				syncTrainingSamplePrompts(filePath);
				syncTrainingSamplePromptsText(content);
				lastSavedSignature = `${filePath}\n${content.trim() ? content : fileContent}`;
				inlineSaveSignature = content;
			}
		} catch {
			// file may not exist yet
		}
		loading = false;
	}

	async function persistFile({ showMessage = true, silent = false } = {}) {
		if (!filePath) {
			if (!silent) error = 'Please set a file path first';
			return;
		}
		if (!filePath.endsWith('.txt')) {
			if (!silent) error = 'File path must end with .txt';
			return;
		}

		saving = true;
		error = '';
		try {
			const res = await fetch('/api/fs/write-file', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ path: filePath, content: fileContent })
			});
			if (!res.ok) {
				const err = await res.json();
				throw new Error(err.detail || 'Failed to save');
			}
			syncTrainingSamplePrompts(filePath);
			syncTrainingSamplePromptsText(fileContent);
			lastSavedSignature = `${filePath}\n${fileContent}`;
			inlineSaveSignature = fileContent;
			success = showMessage ? 'Saved' : 'Autosaved';
			setTimeout(() => {
				if (success === 'Saved' || success === 'Autosaved') success = '';
			}, 1500);
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
			<p class="text-[12px] mt-0.5" style="color: var(--text-muted);">Prompts for periodic sample generation during training. They are stored in the project automatically; a `.txt` file is optional.</p>
		</div>

		<div class="flex items-end gap-2">
			<div class="flex-1">
				<PathInput label="Prompts File" bind:value={filePath} showFiles tooltip="Optional path to import from or export to a sample prompts text file (.txt)." />
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

						<label class="block mb-3">
							<span class="block text-[10px] font-medium mb-1" style="color: var(--text-muted);">Prompt</span>
							<textarea
								class="w-full text-[12px] px-3 py-2 resize-y"
								rows="2"
								value={entry.prompt}
								oninput={(e) => updateEntry(i, 'prompt', e.target.value)}
								placeholder="A cinematic shot of a mountain landscape at sunset, golden hour lighting"
								style="background: var(--console-bg); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none; box-shadow: inset 0 1px 3px rgba(0,0,0,.2);"
								onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
								onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
							></textarea>
						</label>

						<div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-2">
							<FormField label="Width" type="number" value={entry.w} oninput={(e) => updateEntry(i, 'w', Number(e.target.value))} min={64} step={64} tooltip="Output width" />
							<FormField label="Height" type="number" value={entry.h} oninput={(e) => updateEntry(i, 'h', Number(e.target.value))} min={64} step={64} tooltip="Output height" />
							<FormField label="Frames" type="number" value={entry.f} oninput={(e) => updateEntry(i, 'f', Number(e.target.value))} min={1} tooltip="Number of frames" />
							<FormField label="Seed" type="number" value={entry.d} oninput={(e) => updateEntry(i, 'd', e.target.value ? Number(e.target.value) : '')} tooltip="Random seed" />
							<FormField label="Steps" type="number" value={entry.s} oninput={(e) => updateEntry(i, 's', Number(e.target.value))} min={1} tooltip="Sampling steps" />
							<FormField label="Guidance" type="number" value={entry.g} oninput={(e) => updateEntry(i, 'g', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance scale" />
						</div>

						<div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-2">
							<FormField label="CFG Scale" type="number" value={entry.l} oninput={(e) => updateEntry(i, 'l', e.target.value ? Number(e.target.value) : '')} step="0.1" placeholder="None" tooltip="CFG scale (optional)" />
							<FormField label="Flow Shift" type="number" value={entry.fs} oninput={(e) => updateEntry(i, 'fs', Number(e.target.value))} step="0.1" tooltip="Flow shift" />
							<div class="col-span-1 sm:col-span-4">
								<FormField label="Negative" value={entry.n} oninput={(e) => updateEntry(i, 'n', e.target.value)} placeholder="Optional negative prompt" tooltip="Negative prompt" />
							</div>
						</div>

						<div>
							<FormField label="I2V Image (optional)" value={entry.i} oninput={(e) => updateEntry(i, 'i', e.target.value)} placeholder="path/to/first_frame.jpg" tooltip="Image-to-Video: path to conditioning image for first frame" />
						</div>
					</div>
				{/each}
			</div>

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

		<div class="flex items-center gap-3">
			<button
				onclick={() => persistFile({ showMessage: true })}
				disabled={saving || !filePath || promptCount === 0}
				class="px-5 py-2 text-[12px] font-semibold disabled:opacity-40"
				style="background: var(--accent-muted); border: 1px solid var(--accent); color: var(--accent); border-radius: var(--radius-sm);"
				onmouseenter={(e) => { if (!e.currentTarget.disabled) e.currentTarget.style.filter = 'brightness(1.15)'; }}
				onmouseleave={(e) => { e.currentTarget.style.filter = ''; }}
			>{saving ? 'Saving...' : 'Save File'}</button>
			<span class="text-[11px] tabular-nums" style="color: var(--text-muted);">{promptCount} prompt{promptCount !== 1 ? 's' : ''}</span>
			{#if !filePath}
				<span class="text-[11px]" style="color: var(--text-muted);">Stored in project config</span>
			{/if}
			{#if success}
				<span class="text-[11px] font-medium" style="color: var(--success);">{success}</span>
			{/if}
			{#if error}
				<span class="text-[11px]" style="color: var(--danger);">{error}</span>
			{/if}
		</div>

		{#if fileContent}
			<div class="space-y-2">
				<div class="text-[11px] font-medium" style="color: var(--text-muted);">Preview</div>
				<pre class="text-[11px] px-3 py-3 overflow-x-auto" style="background: var(--console-bg); border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-md);">{fileContent}</pre>
			</div>
		{/if}
	</div>
{/if}
