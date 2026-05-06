<script>
	import FieldResetButton from './FieldResetButton.svelte';
	import { labelFromFieldPath } from '$lib/utils/fieldLabels.js';

	let {
		label = '',
		value = $bindable(''),
		options = [],
		placeholder = '',
		tooltip = '',
		disabled = false,
		fieldPath = '',
		oninput,
		...rest
	} = $props();

	let displayLabel = $derived(label || labelFromFieldPath(fieldPath));
	let isOpen = $state(false);
	let inputRef = $state(null);
	let dropdownRef = $state(null);
	let searchQuery = $state('');

	// Filtered options based on search
	let filteredOptions = $derived(
		searchQuery
			? options.filter(opt => opt.toLowerCase().includes(searchQuery.toLowerCase()))
			: options
	);

	function handleInputChange(e) {
		searchQuery = e.target.value;
		value = e.target.value;
		isOpen = true;
		if (oninput) oninput(e);
	}

	function selectOption(option) {
		value = option;
		searchQuery = '';
		isOpen = false;
		if (oninput) {
			const syntheticEvent = { target: { value: option } };
			oninput(syntheticEvent);
		}
	}

	function handleFocus() {
		isOpen = true;
		searchQuery = '';
	}

	function handleBlur(e) {
		// Delay to allow click on dropdown item
		setTimeout(() => {
			if (!dropdownRef?.contains(document.activeElement)) {
				isOpen = false;
				searchQuery = '';
			}
		}, 150);
	}

	function handleKeydown(e) {
		if (e.key === 'Escape') {
			isOpen = false;
			searchQuery = '';
			inputRef?.blur();
		}
	}
</script>

<label class="block relative">
	<span class="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">
		<span data-tooltip={tooltip || undefined}>{displayLabel}</span>
		<FieldResetButton {fieldPath} {disabled} />
	</span>

	<div class="relative">
		<input
			bind:this={inputRef}
			type="text"
			value={isOpen && searchQuery !== '' ? searchQuery : value}
			oninput={handleInputChange}
			onfocus={handleFocus}
			onblur={handleBlur}
			onkeydown={handleKeydown}
			{placeholder}
			{disabled}
			{...rest}
			class="mt-0.5 block w-full px-2.5 py-1.5 pr-8 text-[13px] transition-colors disabled:opacity-40"
			style="background: var(--bg-input); border: 1px solid var(--border); color: var(--text-primary); border-radius: var(--radius-sm);"
		/>

		<!-- Dropdown arrow -->
		<div class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none flex items-center" style="color: var(--text-muted); margin-top: 2px;">
			<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M19 9l-7 7-7-7"/>
			</svg>
		</div>

		<!-- Dropdown menu -->
		{#if isOpen && !disabled}
			<div
				bind:this={dropdownRef}
				class="absolute z-50 w-full mt-1 py-1 max-h-60 overflow-auto"
				style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); box-shadow: var(--shadow-lg);"
			>
				{#if filteredOptions.length > 0}
					{#each filteredOptions as option}
						<button
							type="button"
							onclick={() => selectOption(option)}
							class="w-full px-3 py-1.5 text-left text-[13px] transition-colors"
							style="color: var(--text-primary); background: {value === option ? 'var(--accent-muted)' : 'transparent'};"
							onmouseenter={(e) => e.currentTarget.style.background = 'var(--bg-hover, var(--accent-muted))'}
							onmouseleave={(e) => e.currentTarget.style.background = value === option ? 'var(--accent-muted)' : 'transparent'}
						>
							{option}
						</button>
					{/each}
				{:else if searchQuery}
					<div class="px-3 py-2 text-[12px]" style="color: var(--text-muted);">
						Press Enter to use: <span style="color: var(--text-primary); font-family: var(--font-mono, monospace);">{searchQuery}</span>
					</div>
				{/if}
			</div>
		{/if}
	</div>
</label>
