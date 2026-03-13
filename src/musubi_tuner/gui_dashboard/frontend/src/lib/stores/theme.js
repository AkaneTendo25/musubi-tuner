import { writable } from 'svelte/store';

export const THEMES = [
	{ id: 'warm-terminal', name: 'Warm Terminal', desc: 'Cozy retro-futurism', swatch: '#f0a030' },
	{ id: 'aurora', name: 'Aurora', desc: 'Northern lights', swatch: '#64ffb4' },
	{ id: 'matcha', name: 'Matcha Latte', desc: 'Japanese minimal', swatch: '#5b8c5a' },
	{ id: 'brutalist', name: 'Neon Brutalist', desc: 'Cyberpunk raw', swatch: '#c8ff00' },
	{ id: 'twilight', name: 'Twilight Garden', desc: 'Soft & dreamy', swatch: '#ffb088' }
];

const THEME_IDS = THEMES.map((t) => t.id);
const stored = typeof localStorage !== 'undefined' ? localStorage.getItem('theme') : null;
const initial = stored && THEME_IDS.includes(stored) ? stored : 'warm-terminal';

export const theme = writable(initial);

theme.subscribe((val) => {
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem('theme', val);
	}
	if (typeof document !== 'undefined') {
		const root = document.documentElement;
		THEME_IDS.forEach((id) => root.classList.remove(`theme-${id}`));
		if (val !== 'warm-terminal') {
			root.classList.add(`theme-${val}`);
		}
	}
});

export function setTheme(id) {
	if (THEME_IDS.includes(id)) {
		theme.set(id);
	}
}
