<script>
	import PathInput from './PathInput.svelte';

	let {
		label,
		value = '',
		onchange,
		placeholder = '',
		tooltip = '',
		disabled = false,
		showFiles = false,
		invalid = false,
		error = ''
	} = $props();

	function fireChange(val) {
		if (onchange) onchange(val);
	}

	function handleBrowseSelect(path) {
		fireChange(path);
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div oninput={(e) => { if (e.target?.tagName === 'INPUT') fireChange(e.target.value); }}>
	<PathInput {label} {value} {placeholder} {tooltip} {disabled} {showFiles} {invalid} {error} onselect={handleBrowseSelect} />
</div>
