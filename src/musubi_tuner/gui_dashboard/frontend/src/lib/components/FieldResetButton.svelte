<script>
	import {
		configValuesEqual,
		formatDefaultValue,
		getConfigPathValue,
		projectConfig,
		projectDefaults,
		resetConfigField
	} from '$lib/stores/project.js';

	let { fieldPath = '', disabled = false } = $props();

	let defaultValue = $derived(getConfigPathValue($projectDefaults, fieldPath));
	let currentValue = $derived(getConfigPathValue($projectConfig, fieldPath));
	let changed = $derived(Boolean(fieldPath) && defaultValue !== undefined && !configValuesEqual(currentValue, defaultValue));
	let tooltip = $derived(`Reset to default: ${formatDefaultValue(defaultValue)}`);

	function reset(e) {
		e.preventDefault();
		e.stopPropagation();
		resetConfigField(fieldPath);
	}
</script>

<span class="inline-flex h-4 w-4 flex-shrink-0 items-center justify-center">
	{#if changed}
		<button
			type="button"
			onclick={reset}
			{disabled}
			aria-label="Reset to default"
			data-tooltip={tooltip}
			class="inline-flex h-3.5 w-3.5 items-center justify-center opacity-85 transition-colors hover:opacity-100 disabled:opacity-30"
			style="background: var(--accent-muted); border: 1px solid var(--accent-dim, var(--accent)); color: var(--accent); border-radius: var(--radius-sm);"
			onmouseenter={(e) => { e.currentTarget.style.background = 'var(--accent)'; e.currentTarget.style.color = 'var(--bg-base)'; }}
			onmouseleave={(e) => { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; }}
		>
			<svg class="h-2.5 w-2.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
				<path d="M3 12a9 9 0 1 0 3-6.7" />
				<path d="M3 4v5h5" />
			</svg>
		</button>
	{/if}
</span>
