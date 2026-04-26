export const DEFAULT_MODEL_DIR_NAME = 'models';
export const DEFAULT_LTX2_CHECKPOINT_NAME = 'ltx-2.3-22b-dev.safetensors';
export const DEFAULT_GEMMA_ROOT_NAME = 'gemma-3-12b-it-qat-q4_0-unquantized';

function normalizeBase(path) {
	return (path || '').replace(/[\\/]+$/, '');
}

export function joinPath(base, leaf) {
	const normalized = normalizeBase(base);
	if (!normalized) return leaf;
	const separator = normalized.includes('\\') && !normalized.includes('/') ? '\\' : '/';
	return `${normalized}${separator}${leaf}`;
}

export function defaultModelDir(cwd, config) {
	return config?.model_dir || (cwd ? joinPath(cwd, DEFAULT_MODEL_DIR_NAME) : DEFAULT_MODEL_DIR_NAME);
}

export function effectiveLtx2Checkpoint(cwd, config, explicit = '') {
	return explicit || config?.default_ltx2_checkpoint || joinPath(defaultModelDir(cwd, config), DEFAULT_LTX2_CHECKPOINT_NAME);
}

export function effectiveGemmaRoot(cwd, config, explicit = '', gemmaSafetensors = '') {
	if (gemmaSafetensors) return '';
	return explicit || config?.default_gemma_root || joinPath(defaultModelDir(cwd, config), DEFAULT_GEMMA_ROOT_NAME);
}

export function effectiveGemmaSafetensors(config, explicit = '') {
	return explicit || config?.default_gemma_safetensors || '';
}
