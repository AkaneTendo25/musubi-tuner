import { writable, get } from 'svelte/store';

export const projectConfig = writable(null);
export const projectDefaults = writable({});
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

export function restoreRecentProject(project, index = 0) {
	if (!project?.path) return;
	recentProjects.update((list) => {
		const filtered = list.filter((p) => p.path !== project.path);
		const safeIndex = Math.max(0, Math.min(index, filtered.length));
		const next = [...filtered];
		next.splice(safeIndex, 0, {
			name: project.name || project.path,
			path: project.path,
			lastOpened: project.lastOpened || Date.now(),
		});
		return next.slice(0, 10);
	});
}

export async function closeProject() {
	projectConfig.set(null);
	projectLoaded.set(false);
	projectPath.set(null);
	try { await fetch('/api/project', { method: 'DELETE' }); } catch {}
}

let _saveTimer = null;

function cloneValue(value) {
	if (value === undefined) return undefined;
	if (value === null || typeof value !== 'object') return value;
	return typeof structuredClone === 'function' ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

export function getConfigPathValue(config, path) {
	if (!config || !path) return undefined;
	return path.split('.').reduce((value, key) => value?.[key], config);
}

function cloneContainer(value, nextPart) {
	if (Array.isArray(value)) return [...value];
	if (value && typeof value === 'object') return { ...value };
	return /^\d+$/.test(nextPart) ? [] : {};
}

function setConfigPathValue(config, path, value) {
	const parts = path.split('.');
	if (!config || parts.length === 0) return config;
	const next = Array.isArray(config) ? [...config] : { ...config };
	let cursor = next;
	for (let i = 0; i < parts.length - 1; i += 1) {
		const part = parts[i];
		cursor[part] = cloneContainer(cursor[part], parts[i + 1]);
		cursor = cursor[part];
	}
	cursor[parts[parts.length - 1]] = cloneValue(value);
	return next;
}

export function configValuesEqual(a, b) {
	if (a === undefined && b === undefined) return true;
	if ((a === null || a === undefined) && (b === null || b === undefined)) return true;
	return JSON.stringify(a) === JSON.stringify(b);
}

export function formatDefaultValue(value) {
	if (value === undefined) return 'not set';
	if (value === null) return 'null';
	if (value === '') return 'empty';
	if (typeof value === 'string') return value;
	if (typeof value === 'boolean') return value ? 'true' : 'false';
	return JSON.stringify(value);
}

export async function loadProjectDefaults() {
	try {
		const res = await fetch('/api/project/defaults');
		if (res.ok) {
			const data = await res.json();
			projectDefaults.set(data.config || {});
		}
	} catch {
		// ignore
	}
}

export function resetConfigField(path) {
	const defaults = get(projectDefaults);
	const defaultValue = getConfigPathValue(defaults, path);
	if (defaultValue === undefined) return;
	projectConfig.update((config) => setConfigPathValue(config, path, defaultValue));
	saveProjectDebounced();
}

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
	console.log(`[updateSection] ${section}.${key} =`, value);
	projectConfig.update((c) => {
		if (!c) return c;
		const updated = { ...c, [section]: { ...(c[section] || {}), [key]: value } };
		console.log('[updateSection] Store updated');
		return updated;
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

export async function replaceProjectConfig(config) {
	if (_saveTimer) {
		clearTimeout(_saveTimer);
		_saveTimer = null;
	}
	const res = await fetch('/api/project', {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(config)
	});
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || 'Failed to import settings');
	}
	const data = await res.json();
	projectConfig.set(data.config);
	projectLoaded.set(true);
	const path = get(projectPath);
	if (data.config?.name && path) addRecentProject(data.config.name, path);
	return data.config;
}
