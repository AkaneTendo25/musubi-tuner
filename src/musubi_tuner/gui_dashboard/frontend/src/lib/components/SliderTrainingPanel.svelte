<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import FieldResetButton from '$lib/components/FieldResetButton.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import { projectConfig, saveProjectDebounced } from '$lib/stores/project.js';
	import { processLogs, preloadLogsIfActive, startLogPolling } from '$lib/stores/processes.js';
	import { onMount } from 'svelte';

	onMount(() => {
		preloadLogsIfActive(['slider_training']);
		const logInterval = startLogPolling(['slider_training'], 1000);
		return () => clearInterval(logInterval);
	});

	function update(key, value) {
		projectConfig.update((c) => c ? { ...c, slider: { ...(c.slider || {}), [key]: value } } : c);
		saveProjectDebounced();
	}
	function updateTarget(index, key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			const targets = [...(c.slider?.targets || [{}])];
			targets[index] = { ...(targets[index] || {}), [key]: value };
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}
	function addTarget() {
		projectConfig.update((c) => c ? { ...c, slider: { ...(c.slider || {}), targets: [...(c.slider?.targets || []), { positive: '', negative: '', target_class: '', weight: 1.0 }] } } : c);
		saveProjectDebounced();
	}
	function removeTarget(index) {
		projectConfig.update((c) => {
			if (!c?.slider?.targets) return c;
			const remaining = c.slider.targets.filter((_, i) => i !== index);
			const targets = remaining.length ? remaining : [{ positive: '', negative: '', target_class: '', weight: 1.0 }];
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}
	function updateAnchor(index, key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			const anchors = [...(c.slider?.anchors || [])];
			anchors[index] = { ...(anchors[index] || {}), [key]: value };
			return { ...c, slider: { ...(c.slider || {}), anchors } };
		});
		saveProjectDebounced();
	}
	function addAnchor() {
		projectConfig.update((c) => c ? { ...c, slider: { ...(c.slider || {}), anchors: [...(c.slider?.anchors || []), { prompt: '' }] } } : c);
		saveProjectDebounced();
	}
	function removeAnchor(index) {
		projectConfig.update((c) => c ? { ...c, slider: { ...(c.slider || {}), anchors: (c.slider?.anchors || []).filter((_, i) => i !== index) } } : c);
		saveProjectDebounced();
	}

	let targets = $derived($projectConfig?.slider?.targets || [{ positive: '', negative: '', target_class: '', weight: 1.0 }]);
	let anchors = $derived($projectConfig?.slider?.anchors || []);
	let isH3 = $derived(($projectConfig?.training?.model_type || 'minimax_h3') === 'minimax_h3');
	let sliderLogs = $derived($processLogs.slider_training || []);
</script>

<!-- Slider LoRA -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--accent); opacity: 0.5;"></div>

			<!-- Header -->
			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Slider LoRA</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Train controllable sliders from prompt pairs, paired media caches, or shared reference conditioning</div>
					</div>
				</div>
			</div>

			<!-- Config -->
			<div class="p-5 space-y-4">
				<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
					Model, LoRA, optimizer, memory, and output settings are inherited from the Training tab. Slider-specific cache paths and mode selection live here.
				</p>

				<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
					<!-- Left: Slider settings -->
					<div class="space-y-3">
						<FormGroup title="Slider Settings">
							<div class="space-y-2 pt-2">
								<FormSelect
									fieldPath="slider.mode"
									label="Mode"
									value={$projectConfig?.slider?.mode || 'text'}
									options={isH3
										? [
											{ value: 'text', label: 'text prompts' },
											{ value: 'reference', label: 'paired media' },
											{ value: 'ref2va', label: 'paired media + shared Ref2VA' }
										]
										: [
											{ value: 'text', label: 'text' },
											{ value: 'reference', label: 'reference' },
											{ value: 'ic_reference', label: 'ic_reference (v2v)' }
										]}
									onchange={(e) => update('mode', e.target.value)}
									tooltip="Slider training mode. ic_reference currently reuses the v2v IC-LoRA path."
								/>
								<div class="grid grid-cols-2 gap-2">
									<FormField type="number" fieldPath="slider.max_train_steps" value={$projectConfig?.slider?.max_train_steps ?? 500} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Slider training steps (typically less than full training)" />
									<FormField fieldPath="slider.output_name" value={$projectConfig?.slider?.output_name || 'h3_slider'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix for slider LoRA" />
								</div>
								{#if ($projectConfig?.slider?.mode || 'text') === 'text'}
									<div class="grid grid-cols-2 gap-2">
										{#if isH3}
											<FormField type="number" fieldPath="slider.h3_guidance_strength" value={$projectConfig?.slider?.h3_guidance_strength ?? 1.0} oninput={(e) => update('h3_guidance_strength', Number(e.target.value))} step="0.1" min={0.1} tooltip="Strength of the frozen H3 positive-minus-negative prediction direction" />
										{:else}
											<FormField type="number" fieldPath="slider.guidance_strength" value={$projectConfig?.slider?.guidance_strength ?? 1.0} oninput={(e) => update('guidance_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance strength for text-mode training" />
										{/if}
										<FormField type="number" fieldPath="slider.anchor_strength" value={$projectConfig?.slider?.anchor_strength ?? 1.0} oninput={(e) => update('anchor_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Weight on the anchor preservation loss (text mode only). Anchors keep listed concepts unchanged by the slider." />
									</div>
									{#if !isH3}<div class="grid grid-cols-2 gap-2">
										<FormField type="number" fieldPath="slider.anchor_cap_mult" value={$projectConfig?.slider?.anchor_cap_mult ?? 5.0} oninput={(e) => update('anchor_cap_mult', Number(e.target.value))} step="1" min={0} tooltip="Caps per-step anchor loss at this multiple of its running median (0=off). Tames the rare spikes that inflate grad_norm. Lower (3-5) clamps harder." />
									</div>{/if}
									<div class="grid grid-cols-3 gap-2">
										<FormField fieldPath="slider.latent_frames" label="Frames" type="number" value={$projectConfig?.slider?.latent_frames ?? 1} oninput={(e) => update('latent_frames', Number(e.target.value))} min={1} tooltip="Latent frames (1=image, >1=video)" />
										<FormField type="number" fieldPath="slider.latent_height" value={$projectConfig?.slider?.latent_height ?? 512} oninput={(e) => update('latent_height', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent height" />
										<FormField type="number" fieldPath="slider.latent_width" value={$projectConfig?.slider?.latent_width ?? 768} oninput={(e) => update('latent_width', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent width" />
									</div>
									{#if isH3}
										<div class="grid grid-cols-2 gap-2">
											<FormSelect fieldPath="slider.target_modality" label="Target modality" value={$projectConfig?.slider?.target_modality || 'video'} options={['video', 'audio', 'av']} onchange={(e) => update('target_modality', e.target.value)} tooltip="Train the slider on H3 video, audio, or both outputs" />
											{#if ['audio', 'av'].includes($projectConfig?.slider?.target_modality || 'video')}
												<FormField fieldPath="slider.h3_audio_latent_frames" label="Audio latent frames" type="number" value={$projectConfig?.slider?.h3_audio_latent_frames ?? 81} oninput={(e) => update('h3_audio_latent_frames', Number(e.target.value))} min={1} tooltip="Synthetic audio length for text-mode H3 sliders" />
											{/if}
										</div>
									{/if}
									<FormToggle fieldPath="slider.batch_all_targets" checked={$projectConfig?.slider?.batch_all_targets ?? false} onchange={(e) => update('batch_all_targets', e.target.checked)} tooltip="Process ALL slider targets every step (gradients averaged into one update) instead of picking one target at random per step. Produces a more stable, context-general direction; ~Nx slower per step but visits every target each step." />
								{:else}
									<div class="grid grid-cols-2 gap-2">
										<PathInput fieldPath="slider.pos_cache_dir" value={$projectConfig?.slider?.pos_cache_dir || ''} oninput={(e) => update('pos_cache_dir', e.target.value)} showFiles tooltip="Directory with positive latent caches" />
										<PathInput fieldPath="slider.neg_cache_dir" value={$projectConfig?.slider?.neg_cache_dir || ''} oninput={(e) => update('neg_cache_dir', e.target.value)} showFiles tooltip="Directory with negative latent caches" />
									</div>
								<div class="grid grid-cols-2 gap-2">
									{#if isH3}
										<FormSelect fieldPath="slider.target_modality" value={$projectConfig?.slider?.target_modality || 'video'} options={['video', 'audio', 'av']} onchange={(e) => update('target_modality', e.target.value)} tooltip="Paired H3 target modality" />
									{:else}
										<PathInput fieldPath="slider.text_cache_dir" value={$projectConfig?.slider?.text_cache_dir || ''} oninput={(e) => update('text_cache_dir', e.target.value)} showFiles tooltip="Directory with matching text embedding caches" />
									{/if}
									{#if !isH3 && ($projectConfig?.slider?.mode || 'text') === 'reference'}
										<FormSelect fieldPath="slider.reference_modality" value={$projectConfig?.slider?.reference_modality || 'video'} options={['video', 'audio']} onchange={(e) => update('reference_modality', e.target.value)} tooltip="Paired slider target modality" />
									{:else if ($projectConfig?.slider?.mode || 'text') !== 'reference'}
										<PathInput fieldPath="slider.reference_cache_dir" value={$projectConfig?.slider?.reference_cache_dir || ''} oninput={(e) => update('reference_cache_dir', e.target.value)} showFiles tooltip="Reference latent cache directory used for the shared v2v IC context" />
									{/if}
									</div>
									<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
									{#if isH3 && ($projectConfig?.slider?.mode || 'text') === 'ref2va'}
										Ref2VA mode replaces only the positive/negative target latents; both branches read the same cached reference and Qwen presentation.
									{:else if ($projectConfig?.slider?.mode || 'text') === 'ic_reference'}
										`ic_reference` uses the same cached visual reference clip for both targets.
										{:else}
											Reference sliders train from paired cached examples instead of prompt targets. Use audio modality only with `--ltx2_mode audio`.
										{/if}
									</p>
								{/if}
								<FormField fieldPath="slider.sample_slider_range" value={$projectConfig?.slider?.sample_slider_range || '-2,-1,0,1,2'} oninput={(e) => update('sample_slider_range', e.target.value)} tooltip="Comma-separated multiplier values for preview sampling" />
								<FormField fieldPath="slider.accelerate_extra_args" value={$projectConfig?.slider?.accelerate_extra_args || ''} oninput={(e) => update('accelerate_extra_args', e.target.value)} placeholder="--num_processes 2 --main_process_port 29502" tooltip="Extra arguments appended to `accelerate launch` before the slider training script path." />
								<FormField fieldPath="slider.extra_args" value={$projectConfig?.slider?.extra_args || ''} oninput={(e) => update('extra_args', e.target.value)} placeholder="--flag value --other_flag" tooltip="Extra arguments appended to the slider training script command. Use this for any CLI option without a dedicated dashboard control." />
							</div>
						</FormGroup>
					</div>

					<!-- Right: Targets -->
					<div class="space-y-3">
						<FormGroup title="Slider Targets">
							<div class="space-y-3 pt-2">
								{#if ($projectConfig?.slider?.mode || 'text') === 'text'}
									<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
										Define positive/negative prompt pairs that define the slider direction. The LoRA will learn to move between these attributes.
									</p>
									{#each targets as target, i}
										<div class="p-3 space-y-2 relative" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
											<div class="flex items-center justify-between">
												<span class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--accent);">Target #{i + 1}</span>
												{#if targets.length > 1}
													<button
														onclick={() => removeTarget(i)}
														class="px-2 py-0.5 text-[10px] font-medium"
														style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);"
														onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
														onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
													>
														Remove
													</button>
												{/if}
											</div>
											<!-- svelte-ignore a11y_label_has_associated_control -->
											<label class="block">
												<span class="flex items-center gap-1 text-[10px] font-medium mb-0.5" style="color: var(--success);">
													<span>Positive (+)</span>
													<FieldResetButton fieldPath={`slider.targets.${i}.positive`} />
												</span>
												<textarea
													class="w-full text-[11px] px-2 py-1.5 resize-y"
													rows="2"
													value={target.positive || ''}
													oninput={(e) => updateTarget(i, 'positive', e.target.value)}
													placeholder="high quality, sharp, detailed..."
													style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
													onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
													onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
												></textarea>
											</label>
											<!-- svelte-ignore a11y_label_has_associated_control -->
											<label class="block">
												<span class="flex items-center gap-1 text-[10px] font-medium mb-0.5" style="color: var(--danger);">
													<span>Negative (-)</span>
													<FieldResetButton fieldPath={`slider.targets.${i}.negative`} />
												</span>
												<textarea
													class="w-full text-[11px] px-2 py-1.5 resize-y"
													rows="2"
													value={target.negative || ''}
													oninput={(e) => updateTarget(i, 'negative', e.target.value)}
													placeholder="blurry, low quality, soft..."
													style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
													onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
													onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
												></textarea>
											</label>
											<div class="grid grid-cols-2 gap-2">
												<FormField fieldPath={`slider.targets.${i}.target_class`} label="Target Class" value={target.target_class || ''} oninput={(e) => updateTarget(i, 'target_class', e.target.value)} placeholder="(all content)" tooltip="Optional: restrict to class" />
												<FormField fieldPath={`slider.targets.${i}.weight`} label="Weight" type="number" value={target.weight ?? 1.0} oninput={(e) => updateTarget(i, 'weight', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for this target" />
											</div>
										</div>
									{/each}
									<button
										onclick={addTarget}
										class="w-full py-1.5 text-[11px] font-medium flex items-center justify-center gap-1"
										style="background: var(--bg-elevated); border: 1px dashed var(--border); color: var(--text-muted); border-radius: var(--radius-sm);"
										onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
										onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
									>
										<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 6v12m6-6H6"/></svg>
										Add Target
									</button>

									<!-- Anchors (concept preservation) -->
									<div class="pt-3 mt-1" style="border-top: 1px solid var(--border-subtle);">
										<div class="text-[11px] font-semibold mb-1" style="color: var(--text-primary);">Anchors</div>
										<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
											Optional concepts to preserve. The slider is constrained to leave these prompts unchanged at both <code>+1</code> and <code>-1</code>, preventing drift on unrelated content. Keep the count small (1-3).
										</p>
										{#each anchors as anchor, i}
											<div class="p-3 mb-2 space-y-2 relative" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
												<div class="flex items-center justify-between">
													<span class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--accent);">Anchor #{i + 1}</span>
													<button
														onclick={() => removeAnchor(i)}
														class="px-2 py-0.5 text-[10px] font-medium"
														style="color: var(--text-muted); background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);"
														onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)'; }}
														onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
													>
														Remove
													</button>
												</div>
												<!-- svelte-ignore a11y_label_has_associated_control -->
												<label class="block">
													<span class="flex items-center gap-1 text-[10px] font-medium mb-0.5" style="color: var(--text-secondary);">
														<span>Preserve Prompt</span>
														<FieldResetButton fieldPath={`slider.anchors.${i}.prompt`} />
													</span>
													<textarea
														class="w-full text-[11px] px-2 py-1.5 resize-y"
														rows="2"
														value={anchor.prompt || ''}
														oninput={(e) => updateAnchor(i, 'prompt', e.target.value)}
														placeholder="a portrait of a person..."
														style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); outline: none;"
														onfocus={(e) => e.currentTarget.style.borderColor = 'var(--accent)'}
														onblur={(e) => e.currentTarget.style.borderColor = 'var(--border)'}
													></textarea>
												</label>
											</div>
										{/each}
										<button
											onclick={addAnchor}
											class="w-full py-1.5 text-[11px] font-medium flex items-center justify-center gap-1"
											style="background: var(--bg-elevated); border: 1px dashed var(--border); color: var(--text-muted); border-radius: var(--radius-sm);"
											onmouseenter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)'; }}
											onmouseleave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
										>
											<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 6v12m6-6H6"/></svg>
											Add Anchor
										</button>
									</div>
								{:else}
									<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
										Reference-based slider modes use paired cached examples instead of prompt targets. The positive and negative samples must share basename-aligned cache files.
									</p>
								{/if}
							</div>
						</FormGroup>
					</div>
				</div>
			</div>
		</div>
		<ProcessConsole lines={sliderLogs} processType="slider_training" />
