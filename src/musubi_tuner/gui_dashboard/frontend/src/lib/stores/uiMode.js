import { derived, writable } from 'svelte/store';

export const UI_MODES = ['basic', 'advanced'];

const stored = typeof localStorage !== 'undefined' ? localStorage.getItem('uiMode') : null;
const initial = stored && UI_MODES.includes(stored) ? stored : 'basic';

export const uiMode = writable(initial);
export const advancedMode = derived(uiMode, ($uiMode) => $uiMode === 'advanced');

uiMode.subscribe((value) => {
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem('uiMode', value);
	}
});

export function setUiMode(value) {
	if (UI_MODES.includes(value)) {
		uiMode.set(value);
	}
}

export function toggleUiMode() {
	uiMode.update((value) => (value === 'advanced' ? 'basic' : 'advanced'));
}
