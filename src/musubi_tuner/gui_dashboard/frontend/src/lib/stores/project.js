import { writable, get } from 'svelte/store';

export const projectConfig = writable(null);
export const projectLoaded = writable(false);
export const projectPath = writable(null); // full path to the .json file

// Recent projects (localStorage-backed)
function loadRecentFromStorage() {
	try {
		return JSON.parse(localStorage.getItem('recentProjects') || '[]');
	} catch { return []; }
}
export const recentProjects = writable(loadRecentFromStorage());
recentProjects.subscribe((val) => {
	try { localStorage.setItem('recentProjects', JSON.stringify(val)); } catch {}
});

export function addRecentProject(name, path) {
	recentProjects.update((list) => {
		const filtered = list.filter((p) => p.path !== path);
		return [{ name, path, lastOpened: Date.now() }, ...filtered].slice(0, 10);
	});
}

export function removeRecentProject(path) {
	recentProjects.update((list) => list.filter((p) => p.path !== path));
}

export async function closeProject() {
	projectConfig.set(null);
	projectLoaded.set(false);
	projectPath.set(null);
	try { await fetch('/api/project', { method: 'DELETE' }); } catch {}
}

let _saveTimer = null;

export async function loadProject() {
	try {
		const res = await fetch('/api/project');
		if (res.ok) {
			const data = await res.json();
			if (data.loaded) {
				projectConfig.set(data.config);
				projectLoaded.set(true);
				projectPath.set(data.project_path);
				if (data.config.name && data.project_path) {
					addRecentProject(data.config.name, data.project_path);
				}
				return true;
			}
		}
	} catch {
		// ignore
	}
	return false;
}

export async function createProject(config) {
	const res = await fetch('/api/project', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(config)
	});
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || 'Failed to create project');
	}
	const data = await res.json();
	projectConfig.set(data.config);
	projectLoaded.set(true);
	projectPath.set(data.project_path);
	addRecentProject(data.config.name || config.name, data.project_path);
	return data.config;
}

export async function loadProjectFromPath(path) {
	const res = await fetch('/api/project/load', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ path })
	});
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || 'Failed to load project');
	}
	const data = await res.json();
	projectConfig.set(data.config);
	projectLoaded.set(true);
	projectPath.set(data.project_path);
	addRecentProject(data.config.name, data.project_path);
	return data.config;
}

/**
 * Immutably update a section of the project config.
 * Returns a new object so Svelte's writable store notifies subscribers.
 * Usage: updateSection('training', 'crepa', true)
 *        updateSection('caching', 'skip_existing', false)
 */
export function updateSection(section, key, value) {
	projectConfig.update((c) => {
		if (!c) return c;
		return { ...c, [section]: { ...(c[section] || {}), [key]: value } };
	});
	saveProjectDebounced();
}

export function saveProjectDebounced() {
	if (_saveTimer) clearTimeout(_saveTimer);
	_saveTimer = setTimeout(async () => {
		const config = get(projectConfig);
		if (!config) return;
		try {
			await fetch('/api/project', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(config)
			});
		} catch {
			// ignore
		}
	}, 1000);
}

export async function saveProjectNow() {
	if (_saveTimer) {
		clearTimeout(_saveTimer);
		_saveTimer = null;
	}
	const config = get(projectConfig);
	if (!config) return;
	const res = await fetch('/api/project', {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(config)
	});
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || 'Failed to save project');
	}
}
