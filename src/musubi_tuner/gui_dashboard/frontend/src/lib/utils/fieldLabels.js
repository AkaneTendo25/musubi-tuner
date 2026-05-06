const ACRONYMS = new Map([
	['av', 'AV'],
	['awq', 'AWQ'],
	['cfg', 'CFG'],
	['cts', 'CTS'],
	['cuda', 'CUDA'],
	['cudnn', 'cuDNN'],
	['dcr', 'DCR'],
	['ddp', 'DDP'],
	['dino', 'DINO'],
	['dit', 'DiT'],
	['dop', 'DOP'],
	['ema', 'EMA'],
	['ffn', 'FFN'],
	['fp8', 'FP8'],
	['fps', 'FPS'],
	['hf', 'HF'],
	['hfato', 'HFATO'],
	['i2v', 'I2V'],
	['ic', 'IC'],
	['lr', 'LR'],
	['lokr', 'LoKr'],
	['lora', 'LoRA'],
	['ltx2', 'LTX-2'],
	['nf4', 'NF4'],
	['numpy', 'NumPy'],
	['ogm', 'OGM'],
	['stg', 'STG'],
	['tarp', 'TARP'],
	['tb', 'TB'],
	['tf32', 'TF32'],
	['vae', 'VAE'],
	['v2a', 'V2A'],
	['v2v', 'V2V'],
	['vr', 'VR'],
	['w8a8', 'W8A8'],
]);

function titleWord(word) {
	const lower = word.toLowerCase();
	if (ACRONYMS.has(lower)) return ACRONYMS.get(lower);
	return lower.charAt(0).toUpperCase() + lower.slice(1);
}

export function labelFromFieldPath(fieldPath) {
	const key = (fieldPath || '').split('.').pop() || '';
	if (!key) return '';
	return key.split('_').filter(Boolean).map(titleWord).join(' ');
}
