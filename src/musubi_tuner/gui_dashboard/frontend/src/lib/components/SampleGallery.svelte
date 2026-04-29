<script>
	import { updateTick } from '../stores/sse.js';

	let events = $state([]);
	let samples = $state([]);
	let viewerSrc = $state(null);
	let viewerType = $state('image');

	async function loadSamples() {
		try {
			const res = await fetch('/data/events.json');
			if (!res.ok || res.status === 204) return;
			const allEvents = await res.json();
			const sampleEvents = allEvents.filter((e) => e.type === 'sample' || e.type === 'epoch_sample');

			// Get unique steps, newest first
			const steps = [...new Set(sampleEvents.map((e) => e.step))].sort((a, b) => b - a);
			events = steps;

			// Try to load sample files for each step
			const found = [];
			for (const step of steps.slice(0, 20)) {
				// Common naming patterns for sample files
				const paddedStep = String(step).padStart(6, '0');
				for (const ext of ['mp4', 'png', 'jpg', 'wav']) {
					const url = `/data/samples/${paddedStep}.${ext}`;
					try {
						const check = await fetch(url, { method: 'HEAD' });
						if (check.ok) {
							found.push({ step, url, type: ext === 'mp4' ? 'video' : ext === 'wav' ? 'audio' : 'image' });
						}
					} catch {
						// ignore
					}
				}
				// Also try with prompt index pattern
				for (const idx of [0, 1, 2, 3]) {
					for (const ext of ['mp4', 'png', 'jpg']) {
						const url = `/data/samples/${paddedStep}_${idx}.${ext}`;
						try {
							const check = await fetch(url, { method: 'HEAD' });
							if (check.ok) {
								found.push({ step, url, type: ext === 'mp4' ? 'video' : 'image' });
							}
						} catch {
							// ignore
						}
					}
				}
			}
			samples = found;
		} catch {
			// ignore
		}
	}

	function openViewer(src, type) {
		viewerSrc = src;
		viewerType = type;
	}

	function closeViewer() {
		viewerSrc = null;
	}

	$effect(() => {
		$updateTick;
		loadSamples();
	});
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<h3 class="text-sm font-medium text-gray-400 mb-3">Samples</h3>

	{#if samples.length === 0}
		<p class="text-sm text-gray-600">No samples generated yet</p>
	{:else}
		<div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
			{#each samples as sample}
				<button
					class="relative aspect-video bg-gray-800 rounded overflow-hidden border border-gray-700
							hover:border-blue-500 transition-colors cursor-pointer group"
					onclick={() => openViewer(sample.url, sample.type)}
				>
					{#if sample.type === 'video'}
						<video
							src={sample.url}
							class="w-full h-full object-cover"
							muted
							preload="metadata"
						></video>
						<div class="absolute top-1 right-1 bg-black/60 text-[10px] px-1.5 py-0.5 rounded text-gray-300">
							video
						</div>
					{:else if sample.type === 'audio'}
						<div class="flex items-center justify-center h-full text-gray-500">
							<svg xmlns="http://www.w3.org/2000/svg" class="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
							</svg>
						</div>
					{:else}
						<img src={sample.url} alt="Step {sample.step}" class="w-full h-full object-cover" />
					{/if}
					<div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent
								p-1.5 text-[11px] text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
						Step {sample.step}
					</div>
				</button>
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
			<img src={viewerSrc} alt="Sample" class="max-w-[90vw] max-h-[90vh] object-contain" />
		{/if}
	</button>
{/if}
