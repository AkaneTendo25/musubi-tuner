<script>
	let { processType, status = { state: 'idle', exit_code: null }, onStart, onStop } = $props();

	let loading = $state(false);
	let error = $state('');

	const isRunning = $derived(status.state === 'running' || status.state === 'stopping');
	const canStart = $derived(status.state !== 'running' && status.state !== 'stopping');

	async function handleStart() {
		loading = true;
		error = '';
		try {
			await onStart?.();
		} catch (e) {
			error = e.message;
		}
		loading = false;
	}

	async function handleStop() {
		loading = true;
		error = '';
		try {
			await onStop?.();
		} catch (e) {
			error = e.message;
		}
		loading = false;
	}

	function stateLabel(state) {
		if (state === 'idle') return 'Idle';
		if (state === 'running') return 'Running';
		if (state === 'stopping') return 'Stopping...';
		if (state === 'finished') return 'Finished';
		if (state === 'error') return 'Error';
		return state;
	}

	function stateColor(state) {
		if (state === 'running') return 'var(--success)';
		if (state === 'stopping') return 'var(--warning)';
		if (state === 'error') return 'var(--danger)';
		if (state === 'finished') return 'var(--accent)';
		return 'var(--text-muted)';
	}
</script>

<style>
	.running-blink {
		animation: runningBlink 1.8s ease-in-out infinite;
	}

	@keyframes runningBlink {
		0%, 100% {
			opacity: 1;
		}

		50% {
			opacity: 0.45;
		}
	}
</style>

<div class="flex items-center gap-3 flex-wrap">
	{#if canStart}
		<button
			onclick={handleStart}
			disabled={loading}
			class="px-5 py-2 text-[12px] font-semibold disabled:opacity-50"
			style="background: var(--accent); color: var(--bg-base); border-radius: var(--radius-sm); box-shadow: var(--shadow-sm), var(--glow-accent); font-family: var(--font-label);"
			onmouseenter={(e) => { e.currentTarget.style.filter = 'brightness(1.1)'; }}
			onmouseleave={(e) => { e.currentTarget.style.filter = ''; }}
		>
			{loading ? 'Starting...' : 'Start'}
		</button>
	{:else}
		<button
			onclick={handleStop}
			disabled={loading || status.state === 'stopping'}
			class="px-5 py-2 text-[12px] font-semibold disabled:opacity-50"
			style="background: var(--danger); color: white; border-radius: var(--radius-sm); box-shadow: var(--shadow-sm); font-family: var(--font-label);"
			onmouseenter={(e) => { e.currentTarget.style.filter = 'brightness(1.1)'; }}
			onmouseleave={(e) => { e.currentTarget.style.filter = ''; }}
		>
			{status.state === 'stopping' ? 'Stopping...' : 'Stop'}
		</button>
	{/if}

	<div class="flex items-center gap-2">
		<span class="w-2 h-2 rounded-full flex-shrink-0 {status.state === 'running' ? 'running-blink' : ''}" style="background: {stateColor(status.state)}; {status.state === 'running' ? `box-shadow: 0 0 6px ${stateColor(status.state)};` : ''}"></span>
		<span class="text-[12px] font-medium {status.state === 'running' ? 'running-blink' : ''}" style="color: {stateColor(status.state)}; font-family: var(--font-label);">
			{stateLabel(status.state)}
		</span>
		{#if status.exit_code !== null && status.state !== 'running'}
			<span class="text-[11px]" style="color: var(--text-muted);">(exit {status.exit_code})</span>
		{/if}
	</div>

	{#if error}
		<span class="text-[12px] px-2.5 py-1" style="color: var(--danger); background: var(--danger-muted); border-radius: var(--radius-sm);">{error}</span>
	{/if}
</div>
