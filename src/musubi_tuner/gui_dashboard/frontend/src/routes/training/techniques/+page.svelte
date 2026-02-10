<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess } from '$lib/stores/processes.js';

	function update(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, slider: { ...(c.slider || {}), [key]: value } };
		});
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
		projectConfig.update((c) => {
			if (!c) return c;
			const targets = [...(c.slider?.targets || []), { positive: '', negative: '', target_class: '', weight: 1.0 }];
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}

	function removeTarget(index) {
		projectConfig.update((c) => {
			if (!c?.slider?.targets) return c;
			let targets = c.slider.targets.filter((_, i) => i !== index);
			if (targets.length === 0) targets = [{ positive: '', negative: '', target_class: '', weight: 1.0 }];
			return { ...c, slider: { ...(c.slider || {}), targets } };
		});
		saveProjectDebounced();
	}

	function updateTraining(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, training: { ...(c.training || {}), [key]: value } };
		});
		saveProjectDebounced();
	}

	function updateCaching(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, caching: { ...(c.caching || {}), [key]: value } };
		});
		saveProjectDebounced();
	}

	let targets = $derived($projectConfig?.slider?.targets || [{ positive: '', negative: '', target_class: '', weight: 1.0 }]);
	let sliderStatus = $derived($processStatuses.slider_training || { state: 'idle', exit_code: null });
	let dinoStatus = $derived($processStatuses.cache_dino || { state: 'idle', exit_code: null });
	let dinoLogs = $derived($processLogs.cache_dino || []);
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

		<!-- CREPA -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">CREPA</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Cross-frame Representation Alignment (arxiv 2506.09229)</div>
					</div>
					<div class="ml-auto">
						<FormToggle checked={$projectConfig?.training?.crepa ?? false} onchange={(e) => updateTraining('crepa', e.target.checked)} />
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					Aligns intermediate DiT representations across video frames during fine-tuning, improving temporal consistency. A small projector MLP is trained alongside LoRA.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<!-- Teacher signal mode -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Teacher Signal</div>
					<FormSelect label="Mode" value={$projectConfig?.training?.crepa_mode || 'backbone'} onchange={(e) => updateTraining('crepa_mode', e.target.value)} options={[{value: 'backbone', label: 'Backbone (deeper block)'}, {value: 'dino', label: 'DINOv2 (pre-cached)'}]} tooltip="backbone: deeper transformer block as teacher. dino: pre-cached DINOv2 features (zero VRAM, must cache first on Caching tab)." />

					<div class="grid grid-cols-2 gap-2 mt-2">
						<FormField label="Student Block" type="number" value={$projectConfig?.training?.crepa_student_block_idx ?? 16} oninput={(e) => updateTraining('crepa_student_block_idx', Number(e.target.value))} min={0} max={47} tooltip={($projectConfig?.training?.crepa_mode || 'backbone') === 'backbone' ? 'Early block whose hidden states are aligned to the teacher (default 16)' : 'DiT block whose hidden states are projected into DINOv2 space (default 16)'} />
						<FormField label="Teacher Block" type="number" value={$projectConfig?.training?.crepa_teacher_block_idx ?? 32} oninput={(e) => updateTraining('crepa_teacher_block_idx', Number(e.target.value))} min={0} max={47} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'backbone'} tooltip="Deeper block providing the teacher signal (default 32, must be > student)" />
					</div>
					<FormSelect label="DINOv2 Model" value={$projectConfig?.training?.crepa_dino_model || 'dinov2_vitb14'} onchange={(e) => updateTraining('crepa_dino_model', e.target.value)} options={[{value: 'dinov2_vits14', label: 'ViT-S/14 (384d)'}, {value: 'dinov2_vitb14', label: 'ViT-B/14 (768d)'}, {value: 'dinov2_vitl14', label: 'ViT-L/14 (1024d)'}, {value: 'dinov2_vitg14', label: 'ViT-G/14 (1536d)'}]} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'dino'} tooltip="DINOv2 model variant. Must match the model used during caching." />
				</div>

				<!-- Loss parameters -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss Parameters</div>
					<div class="grid grid-cols-3 gap-2 mb-2">
						<FormField label="Lambda" type="number" value={$projectConfig?.training?.crepa_lambda ?? 0.1} oninput={(e) => updateTraining('crepa_lambda', Number(e.target.value))} step="0.01" min={0} tooltip="CREPA loss weight (default 0.1)" />
						<FormField label="Tau" type="number" value={$projectConfig?.training?.crepa_tau ?? 1.0} oninput={(e) => updateTraining('crepa_tau', Number(e.target.value))} step="0.1" min={0.01} tooltip="Temporal neighbor decay factor (default 1.0)" />
						<FormField label="Neighbors" type="number" value={$projectConfig?.training?.crepa_num_neighbors ?? 2} oninput={(e) => updateTraining('crepa_num_neighbors', Number(e.target.value))} min={1} max={8} tooltip="K frames on each side for alignment (default 2)" />
					</div>
					<div class="grid grid-cols-3 gap-2">
						<FormSelect label="Schedule" value={$projectConfig?.training?.crepa_schedule || 'constant'} onchange={(e) => updateTraining('crepa_schedule', e.target.value)} options={[{value: 'constant', label: 'Constant'}, {value: 'linear', label: 'Linear decay'}, {value: 'cosine', label: 'Cosine decay'}]} tooltip="Lambda schedule over training" />
						<FormField label="Warmup Steps" type="number" value={$projectConfig?.training?.crepa_warmup_steps ?? 0} oninput={(e) => updateTraining('crepa_warmup_steps', Number(e.target.value))} min={0} tooltip="Steps before CREPA loss reaches full strength" />
						<div class="flex items-end pb-0.5">
							<FormToggle label="Normalize" checked={$projectConfig?.training?.crepa_normalize ?? true} onchange={(e) => updateTraining('crepa_normalize', e.target.checked)} tooltip="L2-normalize features before similarity computation" />
						</div>
					</div>
				</div>

				<!-- DINOv2 Caching (for CREPA dino mode) -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">DINOv2 Feature Caching</div>
					<p class="text-[10px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Cache DINOv2 features for CREPA dino mode. Run this after latent caching and before training. Uses the DINOv2 model selected above.
					</p>
					<div class="mb-2">
						<FormField label="Batch Size" type="number" value={$projectConfig?.caching?.dino_batch_size ?? 16} oninput={(e) => updateCaching('dino_batch_size', Number(e.target.value))} min={1} disabled={($projectConfig?.training?.crepa_mode || 'backbone') !== 'dino'} tooltip="Frames per DINOv2 forward pass (reduce if OOM)" />
					</div>
					<div class="mb-2">
						<ProcessControls processType="cache_dino" status={dinoStatus} onStart={() => startProcess('cache_dino')} onStop={() => stopProcess('cache_dino')} />
					</div>
					<ProcessConsole lines={dinoLogs} />
					<CommandPanel processType="cache_dino" defaultFilename="cache_dino.bat" />
				</div>
			</div>
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
						<FormToggle checked={$projectConfig?.training?.blank_preservation ?? false} onchange={(e) => updateTraining('blank_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Regularizes by training on blank (empty) prompts alongside real data, preserving the model's base generation capabilities.
					</p>
					<FormField label="Multiplier" type="number" value={$projectConfig?.training?.blank_preservation_multiplier ?? 1.0} oninput={(e) => updateTraining('blank_preservation_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for blank preservation (default 1.0)" />
				</div>

				<!-- DOP -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">DOP (Differential Output Preservation)</span>
						<FormToggle checked={$projectConfig?.training?.dop ?? false} onchange={(e) => updateTraining('dop', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Preserves the model's output distribution for a specified class by penalizing deviations from the original model during training.
					</p>
					<div class="grid grid-cols-2 gap-2">
						<FormField label="Class Prompt" value={$projectConfig?.training?.dop_class || ''} oninput={(e) => updateTraining('dop_class', e.target.value)} placeholder="woman" tooltip="Target class prompt for output preservation" />
						<FormField label="Multiplier" type="number" value={$projectConfig?.training?.dop_multiplier ?? 1.0} oninput={(e) => updateTraining('dop_multiplier', Number(e.target.value))} step="0.1" min={0} tooltip="Loss weight for DOP (default 1.0)" />
					</div>
				</div>

				<!-- Prior Divergence -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Prior Divergence</span>
						<FormToggle checked={$projectConfig?.training?.prior_divergence ?? false} onchange={(e) => updateTraining('prior_divergence', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						KL-divergence regularization that penalizes the trained model from diverging too far from the original pretrained model weights.
					</p>
					<FormField label="Multiplier" type="number" value={$projectConfig?.training?.prior_divergence_multiplier ?? 0.1} oninput={(e) => updateTraining('prior_divergence_multiplier', Number(e.target.value))} step="0.01" min={0} tooltip="KL-divergence regularization strength (default 0.1)" />
				</div>

				<!-- Precaching options -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-1">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Precached Preservation</span>
						<FormToggle checked={$projectConfig?.training?.use_precached_preservation ?? false} onchange={(e) => updateTraining('use_precached_preservation', e.target.checked)} />
					</div>
					<p class="text-[11px] leading-relaxed mb-2" style="color: var(--text-muted);">
						Use pre-cached text encoder outputs for preservation prompts (must be cached during the caching step).
					</p>
					<PathInput label="Cache Dir" value={$projectConfig?.training?.preservation_prompts_cache || ''} oninput={(e) => updateTraining('preservation_prompts_cache', e.target.value)} showFiles tooltip="Directory with cached preservation embeddings" />
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
				<p class="text-[11px] leading-relaxed" style="color: var(--text-muted);">
					Model, LoRA, optimizer, memory, and output settings are inherited from the Training tab. Only slider-specific settings are shown here.
				</p>

				<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
					<!-- Left: Slider settings -->
					<div class="space-y-3">
						<FormGroup title="Slider Settings">
							<div class="space-y-2 pt-2">
								<div class="grid grid-cols-2 gap-2">
									<FormField label="Steps" type="number" value={$projectConfig?.slider?.max_train_steps ?? 500} oninput={(e) => update('max_train_steps', Number(e.target.value))} min={1} tooltip="Slider training steps (typically less than full training)" />
									<FormField label="Output Name" value={$projectConfig?.slider?.output_name || 'ltx2_slider'} oninput={(e) => update('output_name', e.target.value)} tooltip="Output filename prefix for slider LoRA" />
								</div>
								<FormField label="Guidance Strength" type="number" value={$projectConfig?.slider?.guidance_strength ?? 1.0} oninput={(e) => update('guidance_strength', Number(e.target.value))} step="0.1" min={0} tooltip="Guidance strength for text-mode training" />
								<div class="grid grid-cols-3 gap-2">
									<FormField label="Frames" type="number" value={$projectConfig?.slider?.latent_frames ?? 1} oninput={(e) => update('latent_frames', Number(e.target.value))} min={1} tooltip="Latent frames (1=image, >1=video)" />
									<FormField label="Height" type="number" value={$projectConfig?.slider?.latent_height ?? 512} oninput={(e) => update('latent_height', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent height" />
									<FormField label="Width" type="number" value={$projectConfig?.slider?.latent_width ?? 768} oninput={(e) => update('latent_width', Number(e.target.value))} min={64} step={64} tooltip="Synthetic latent width" />
								</div>
								<FormField label="Sample Slider Range" value={$projectConfig?.slider?.sample_slider_range || '-2,-1,0,1,2'} oninput={(e) => update('sample_slider_range', e.target.value)} tooltip="Comma-separated multiplier values for preview sampling" />
							</div>
						</FormGroup>
					</div>

					<!-- Right: Targets -->
					<div class="space-y-3">
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
