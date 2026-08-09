<script>
	import { updateTick } from '../stores/sse.js';

	let samples = $state([]);
	let viewerSrc = $state(null);
	let viewerType = $state('image');
	let viewerAudio = $state(null);

	const SOURCE_LABELS = { i2v: 'I2V ref', v2v: 'V2V ref', refaudio: 'ref audio' };

	function normalizeSources(raw) {
		if (!raw || typeof raw !== 'object') return [];
		const out = [];
		for (const kind of ['i2v', 'v2v', 'refaudio']) {
			const entry = raw[kind];
			if (entry && entry.url) {
				out.push({
					kind,
					label: SOURCE_LABELS[kind] || kind,
					url: entry.url,
					media: entry.media || 'image'
				});
			}
		}
		return out;
	}

	async function loadSamples() {
		try {
			const res = await fetch('/data/samples-index');
			if (!res.ok || res.status === 204) return;
			const data = await res.json();
			const items = Array.isArray(data?.samples) ? data.samples : [];
			samples = items.slice(0, 60).map((item) => ({
				step: item.step,
				promptIdx: item.prompt_idx,
				isEpoch: item.is_epoch,
				url: item.url,
				type: item.kind,
				audioUrl: item.audio_url,
				hasAudio: item.has_audio,
				sources: normalizeSources(item.sources)
			}));
		} catch {
			// ignore
		}
	}

	function openViewer(sample) {
		viewerSrc = sample.url;
		viewerType = sample.type;
		// Only attach a separate audio track for non-video kinds (videos with audio
		// use the muxed _av.mp4 directly); video tiles get their own player anyway.
		viewerAudio = sample.type === 'video' ? null : sample.audioUrl;
	}

	function closeViewer() {
		viewerSrc = null;
		viewerAudio = null;
	}

	$effect(() => {
		$updateTick;
		loadSamples();
	});
</script>

<div class="p-4">
	<h3 class="text-sm font-medium mb-3" style="color: var(--text-secondary);">Samples</h3>

	{#if samples.length === 0}
		<p class="text-sm" style="color: var(--text-muted);">No samples generated yet</p>
	{:else}
		<div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
			{#each samples as sample}
				<div class="flex flex-col gap-1.5">
				<button
					class="relative aspect-video rounded overflow-hidden border transition-colors cursor-pointer group"
					onclick={() => openViewer(sample)}
					style="background: transparent; border-color: var(--border);"
					onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; }}
					onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; }}
				>
					{#if sample.type === 'video'}
						<video
							src={sample.url}
							class="w-full h-full object-cover"
							muted
							preload="metadata"
						></video>
						<div class="absolute top-1 right-1 bg-black/60 text-[10px] px-1.5 py-0.5 rounded text-gray-300">
							{sample.hasAudio ? 'video + audio' : 'video'}
						</div>
					{:else if sample.type === 'audio'}
						<div class="flex items-center justify-center h-full text-gray-500">
							<svg xmlns="http://www.w3.org/2000/svg" class="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
							</svg>
						</div>
					{:else}
						<img src={sample.url} alt="Step {sample.step} prompt {sample.promptIdx}" class="w-full h-full object-cover" />
					{/if}
					<div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent
								p-1.5 text-[11px] text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
						{sample.isEpoch ? 'Epoch' : 'Step'} {sample.step} · #{String(sample.promptIdx).padStart(2, '0')}
					</div>
				</button>
				{#if sample.sources.length > 0}
					<div class="flex items-center gap-1.5 flex-wrap" style="color: var(--text-muted);">
						<span class="text-[10px] uppercase tracking-wider">src:</span>
						{#each sample.sources as src}
							<button
								class="relative w-10 h-10 rounded overflow-hidden border"
								onclick={() => openViewer({ url: src.url, type: src.media, audioUrl: null, hasAudio: src.media === 'audio' })}
								style="background: var(--bg-elevated); border-color: var(--border);"
								onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; }}
								onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; }}
								title="{src.label}"
							>
								{#if src.media === 'image'}
									<img src={src.url} alt={src.label} class="w-full h-full object-cover" />
								{:else if src.media === 'video'}
									<video src={src.url} class="w-full h-full object-cover" muted preload="metadata"></video>
								{:else}
									<div class="flex items-center justify-center h-full">
										<svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
										</svg>
									</div>
								{/if}
							</button>
							<span class="text-[10px]">{src.label}</span>
						{/each}
					</div>
				{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<!-- Fullscreen viewer -->
{#if viewerSrc}
	<button
		class="fixed inset-0 z-50 bg-black/90 flex items-center justify-center cursor-pointer"
		onclick={closeViewer}
		onkeydown={(e) => e.key === 'Escape' && closeViewer()}
	>
		{#if viewerType === 'video'}
			<video src={viewerSrc} class="max-w-[90vw] max-h-[90vh]" controls autoplay>
				<track kind="captions" />
			</video>
		{:else if viewerType === 'audio'}
			<audio src={viewerSrc} controls autoplay></audio>
		{:else}
			<div class="flex flex-col items-center gap-3">
				<img src={viewerSrc} alt="Sample" class="max-w-[90vw] max-h-[80vh] object-contain" />
				{#if viewerAudio}
					<audio src={viewerAudio} controls autoplay></audio>
				{/if}
			</div>
		{/if}
	</button>
{/if}
