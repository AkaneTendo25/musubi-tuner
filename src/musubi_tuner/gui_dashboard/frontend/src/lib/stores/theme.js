import { writable } from 'svelte/store';

export const THEMES = [
	{ id: 'warm-terminal', name: 'Warm Terminal', desc: 'Cozy retro-futurism', swatch: '#f0a030' },
	{ id: 'aurora', name: 'Aurora', desc: 'Northern lights', swatch: '#64ffb4' },
	{ id: 'matcha', name: 'Matcha Latte', desc: 'Japanese minimal', swatch: '#5b8c5a' },
	{ id: 'brutalist', name: 'Neon Brutalist', desc: 'Cyberpunk raw', swatch: '#c8ff00' },
	{ id: 'twilight', name: 'Twilight Garden', desc: 'Soft & dreamy', swatch: '#ffb088' },
	{ id: 'synthwave', name: 'Synthwave', desc: '80s retrowave neon', swatch: '#ff2dca' },
	{ id: 'sakura', name: 'Sakura', desc: 'Cherry blossom light', swatch: '#d4546e' },
	{ id: 'deep-sea', name: 'Deep Sea', desc: 'Bioluminescent abyss', swatch: '#00d4d4' },
	{ id: 'mono', name: 'Mono Slate', desc: 'Minimal monochrome', swatch: '#4a90ff' }
];

const THEME_IDS = THEMES.map((t) => t.id);
const THEME_STORAGE_KEY = 'theme';

function readStoredTheme() {
	if (typeof window === 'undefined') return null;
	try {
		const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
		return stored && THEME_IDS.includes(stored) ? stored : null;
	} catch {
		return null;
	}
}

function persistTheme(id) {
	if (typeof window === 'undefined') return;
	try {
		window.localStorage.setItem(THEME_STORAGE_KEY, id);
	} catch {}
}

function applyThemeClass(id) {
	if (typeof document === 'undefined') return;
	const root = document.documentElement;
	THEME_IDS.forEach((themeId) => root.classList.remove(`theme-${themeId}`));
	if (id !== 'warm-terminal') {
		root.classList.add(`theme-${id}`);
	}
}

const stored = readStoredTheme();
const initial = stored && THEME_IDS.includes(stored) ? stored : 'warm-terminal';

export const theme = writable(initial);

theme.subscribe((val) => {
	persistTheme(val);
	applyThemeClass(val);
});

export function setTheme(id) {
	if (THEME_IDS.includes(id)) {
		theme.set(id);
	}
}
