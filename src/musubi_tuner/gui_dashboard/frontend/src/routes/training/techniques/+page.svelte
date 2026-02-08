<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import CheckpointInput from '$lib/components/CheckpointInput.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';
	import { processStatuses, startProcess, stopProcess } from '$lib/stores/processes.js';

	function update(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			if (!c.slider) c.slider = {};
			c.slider[key] = value;
			return c;
		});
		saveProjectDebounced();
	}

	function updateTarget(index, key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			if (!c.slider) c.slider = {};
			if (!c.slider.targets) c.slider.targets = [{}];
			if (!c.slider.targets[index]) c.slider.targets[index] = {};
			c.slider.targets[index][key] = value;
			return c;
		});
		saveProjectDebounced();
	}

	function addTarget() {
		projectConfig.update((c) => {
			if (!c) return c;
			if (!c.slider) c.slider = {};
			if (!c.slider.targets) c.slider.targets = [];
			c.slider.targets = [...c.slider.targets, { positive: '', negative: '', target_class: '', weight: 1.0 }];
			return c;
		});
		saveProjectDebounced();
	}

	function removeTarget(index) {
		projectConfig.update((c) => {
			if (!c?.slider?.targets) return c;
			c.slider.targets = c.slider.targets.filter((_, i) => i !== index);
			if (c.slider.targets.length === 0) c.slider.targets = [{ positive: '', negative: '', target_class: '', weight: 1.0 }];
			return c;
		});
		saveProjectDebounced();
	}

	function updateTraining(key, value) {
		projectConfig.update((c) => { if (!c) return c; if (!c.training) c.training = {}; c.training[key] = value; return c; });
		saveProjectDebounced();
	}

	let t = $derived($projectConfig?.training || {});
	let s = $derived($projectConfig?.slider || {});
	let targets = $derived(s.targets || [{ positive: '', negative: '', target_class: '', weight: 1.0 }]);
	let sliderStatus = $derived($processStatuses.slider_training || { state: 'idle', exit_code: null });
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else}
	<div class="space-y-5">
		<div>
			<h2 class="text-base font-semibold" style="color: var(--text-primary);">Training Techniques</h2>
			<p class="text-[12px]" style="color: var(--text-muted);">Advanced training enhancements and specialized LoRA types.</p>
		</div>

		<!-- CREPA — coming soon -->
		<div class="p-5" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.4;"></div>
			<div class="flex items-center gap-3 mb-3">
				<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
					<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/></svg>
				</div>
				<div>
					<div class="text-[13px] font-semibold" style="color: var(--text-primary);">CREPA</div>
					<div class="text-[11px]" style="color: var(--text-muted);">Cross-frame Representation Alignment for Fine-tuning Video Diffusion Models</div>
				</div>
				<div class="ml-auto inline-flex items-center gap-1.5 px-2 py-1 text-[10px] font-medium" style="background: var(--bg-elevated); border-radius: var(--radius-full); color: var(--text-muted);">
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
					Coming soon
				</div>
			</div>
			<p class="text-[12px] leading-relaxed" style="color: var(--text-secondary);">
				Aligns intermediate representations across video frames during fine-tuning, improving temporal consistency and reducing flickering artifacts in generated videos.
			</p>
		</div>

		<!-- Preservation & Regularization -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Preservation & Regularization</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Techniques to prevent catastrophic forgetting and maintain model quality</div>
					</div>
				</div>
			</div>

			<div class="p-5 space-y-4">
				<!-- Blank Preservation -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Blank Preservation</span>
						<FormToggle checked={t.blank_preservation ?? false} onchange={(e) => updateTraining('blank_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Regularizes by training on blank (empty) prompts alongside real data, preserving the model's base generation capabilities.
					</p>
					<FormField label="Multiplier" type="number" value={t.blank_preservation_multiplier ?? 1.0} oninput={(e) => updateTraining('blank_preservation_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for blank preservation (default 1.0)" />
				</div>

				<!-- DOP -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">DOP (Differential Output Preservation)</span>
						<FormToggle checked={t.dop ?? false} onchange={(e) => updateTraining('dop', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Preserves the model's output distribution for a specified class by penalizing deviations from the original model during training.
					</p>
					<div class="grid grid-cols-2 gap-2">
						<FormField label="Class Prompt" value={t.dop_class || ''} oninput={(e) => updateTraining('dop_class', e.target.value)} placeholder="woman" tooltip="Target class prompt for output preservation" />
						<FormField label="Multiplier" type="number" value={t.dop_multiplier ?? 1.0} oninput={(e) => updateTraining('dop_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for DOP (default 1.0)" />
					</div>
				</div>

				<!-- Prior Divergence -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Prior Divergence</span>
						<FormToggle checked={t.prior_divergence ?? false} onchange={(e) => updateTraining('prior_divergence', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						KL-divergence regularization that penalizes the trained model from diverging too far from the original pretrained model weights.
					</p>
					<FormField label="Multiplier" type="number" value={t.prior_divergence_multiplier ?? 0.1} oninput={(e) => updateTraining('prior_divergence_multiplier', Number(e.target.value))} step="0.01" min={0} tooltip="KL-divergence regularization strength (default 0.1)" />
				</div>

				<!-- Precaching options -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Precached Preservation</span>
						<FormToggle checked={t.use_precached_preservation ?? false} onchange={(e) => updateTraining('use_precached_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Use pre-cached text encoder outputs for preservation prompts (must be cached during the caching step).
					</p>
					<PathInput label="Cache Dir" value={t.preservation_prompts_cache || ''} oninput={(e) => updateTraining('preservation_prompts_cache', e.target.value)} showFiles tooltip="Directory with cached preservation embeddings" />
				</div>
			</div>
		</div>

		<!-- Slider LoRA — functional -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<!-- Header -->
			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Slider LoRA</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Train controllable attribute sliders with prompt pairs</div>
					</div>
				</div>
			</div>

			<!-- Config -->
			<div class="p-5 space-y-4">
				<!-- Two-column layout -->
				<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
					<!-- Left: Model + Output -->
					<div class="space-y-3">
						<FormGroup title="Model">
							<div class="space-y-2 pt-2">
								<CheckpointInput label="LTX-2 Checkpoint" value={s.ltx2_checkpoint || ''} onchange={(v) => update('ltx2_checkpoint', v)} showFiles scanType="ltx2" tooltip="Path to LTX-2 checkpoint" />
								<CheckpointInput label="Gemma Root" value={s.gemma_root || ''} onchange={(v) => update('gemma_root', v)} scanType="gemma" tooltip="Gemma text encoder directory" />
								<div class="flex flex-wrap gap-x-4 gap-y-1">
									<FormToggle label="FP8 Base" checked={s.fp8_base ?? false} onchange={(e) => update('fp8_base', e.target.checked)} tooltip="FP8 precision" />
									<FormToggle label="Flash Attn" checked={s.flash_attn ?? true} onchange={(e) => update('flash_attn', e.target.checked)} tooltip="Flash attention" />
									<FormToggle label="Gemma 8b" checked={s.gemma_load_in_8bit ?? false} onchange={(e) => update('gemma_load_in_8bit', e.target.checked)} tooltip="8-bit quantization" />
									<FormToggle label="Gemma 4b" checked={s.gemma_load_in_4bit ?? false} onchange={(e) => update('gemma_load_in_4bit', e.target.checked)} tooltip="4-bit quantization" />
								</div>
							</div>
						</FormGroup>

						<FormGroup title="LoRA & Training">
							<div class="space-y-2 pt-2">
								<div class="grid grid-cols-3 gap-2">
									<FormField label="Dim" type="number" value={s.network_dim ?? 16} oninput={(e) => update('network_dim', Number(e.target.value))} min={1} tooltip="LoRA rank" />
									<FormField label="Alpha" type="number" value={s.network_alpha ?? 16} oninput={(e) => update('network_alpha', Number(e.target.value))} min={1} tooltip="LoRA alpha" />
									<FormField label="Steps" type="number" value={s.max_train_steps ?? 500} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Training steps" />
								</div>
								<div class="grid grid-cols-3 gap-2">
									<FormField label="LR" value={s.learning_rate ?? 1e-4} oninput={(e) => update('learning_rate', Number(e.target.value))} tooltip="Learning rate" />
									<FormField label="Optimizer" value={s.optimizer_type || 'adamw8bit'} oninput={(e) => update('optimizer_type', e.target.value)} tooltip="Optimizer type" />
									<FormField label="Grad Accum" type="number" value={s.gradient_accumulation_steps ?? 1} oninput={(e) => update('gradient_accumulation_steps', Number(e.target.value))} min={1} tooltip="Gradient accumulation" />
								</div>
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Blocks to Swap" type="number" value={s.blocks_to_swap ?? ''} oninput={(e) => update('blocks_to_swap', e.target.value ? Number(e.target.value) : null)} placeholder="0" min={0} max={40} tooltip="CPU offload blocks" />
									<FormField label="Seed" type="number" value={s.seed ?? ''} oninput={(e) => update('seed', e.target.value ? Number(e.target.value) : null)} placeholder="Random" tooltip="Random seed" />
								</div>
								<FormToggle label="Gradient Checkpointing" checked={s.gradient_checkpointing ?? true} onchange={(e) => update('gradient_checkpointing', e.target.checked)} tooltip="Save VRAM with checkpointing" />
							</div>
						</FormGroup>

						<FormGroup title="Output">
							<div class="space-y-2 pt-2">
								<FormField label="Output Dir" value={s.output_dir || ''} oninput={(e) => update('output_dir', e.target.value)} placeholder="output" tooltip="Output directory" />
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Name" value={s.output_name || 'ltx2_slider'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix" />
									<FormField label="Save Every N Steps" type="number" value={s.save_every_n_steps ?? ''} oninput={(e) => update('save_every_n_steps', e.target.value ? Number(e.target.value) : null)} placeholder="None" tooltip="Checkpoint interval" />
								</div>
							</div>
						</FormGroup>
					</div>

					<!-- Right: Slider-specific settings + Targets -->
					<div class="space-y-3">
						<FormGroup title="Slider Settings">
							<div class="space-y-2 pt-2">
								<FormField label="Guidance Strength" type="number" value={s.guidance_strength ?? 1.0} oninput={(e) => update('guidance_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance strength for text-mode training" />
								<div class="grid grid-cols-3 gap-2">
									<FormField label="Frames" type="number" value={s.latent_frames ?? 1} oninput={(e) => update('latent_frames', Number(e.target.value))} min={1} tooltip="Latent frames (1=image, >1=video)" />
									<FormField label="Height" type="number" value={s.latent_height ?? 512} oninput={(e) => update('latent_height', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent height" />
									<FormField label="Width" type="number" value={s.latent_width ?? 768} oninput={(e) => update('latent_width', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent width" />
								</div>
								<FormField label="Sample Slider Range" value={s.sample_slider_range || '-2,-1,0,1,2'} oninput={(e) => update('sample_slider_range', e.target.value)} tooltip="Comma-separated multiplier values for preview sampling" />
							</div>
						</FormGroup>

						<!-- Targets editor -->
						<FormGroup title="Slider Targets">
							<div class="space-y-3 pt-2">
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
													class="w-5 h-5 flex items-center justify-center"
													style="color: var(--text-muted); border-radius: var(--radius-sm);"
													onmouseenter={(e) => { e.currentTarget.style.color = 'var(--danger)'; }}
													onmouseleave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
													title="Remove target"
												>
													<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
												</button>
											{/if}
										</div>
										<!-- svelte-ignore a11y_label_has_associated_control -->
										<label class="block">
											<span class="block text-[10px] font-medium mb-0.5" style="color: var(--success);">Positive (+)</span>
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
											<span class="block text-[10px] font-medium mb-0.5" style="color: var(--danger);">Negative (-)</span>
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
											<FormField label="Target Class" value={target.target_class || ''} oninput={(e) => updateTarget(i, 'target_class', e.target.value)} placeholder="(all content)" tooltip="Optional: restrict to class" />
											<FormField label="Weight" type="number" value={target.weight ?? 1.0} oninput={(e) => updateTarget(i, 'weight', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for this target" />
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
							</div>
						</FormGroup>
					</div>
				</div>

				<!-- Process Controls -->
				<div class="pt-2">
					<ProcessControls processType="slider_training" status={sliderStatus} onStart={() => startProcess('slider_training')} onStop={() => stopProcess('slider_training')} />
				</div>

				<CommandPanel processType="slider_training" defaultFilename="slider_train.bat" />
			</div>
		</div>
	</div>
{/if}
