<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import CommandPanel from '$lib/components/CommandPanel.svelte';
	import { projectConfig, projectLoaded, saveProjectDebounced } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, preloadLogsIfActive, startLogPolling } from '$lib/stores/processes.js';
	import { advancedMode } from '$lib/stores/uiMode.js';
	import { onMount } from 'svelte';

	onMount(() => {
		preloadLogsIfActive(['rl_cache_rollouts', 'rl_train']);
		const logInterval = startLogPolling(['rl_cache_rollouts', 'rl_train'], 1000);
		return () => clearInterval(logInterval);
	});

	function update(key, value) {
		projectConfig.update((c) => {
			if (!c) return c;
			return { ...c, rl: { ...(c.rl || {}), [key]: value } };
		});
		saveProjectDebounced();
	}

	let online = $derived($projectConfig?.rl?.online ?? false);
	let rlLoss = $derived($projectConfig?.rl?.rl_loss || 'nft');
	let cacheStatus = $derived($processStatuses.rl_cache_rollouts || { state: 'idle', exit_code: null });
	let trainStatus = $derived($processStatuses.rl_train || { state: 'idle', exit_code: null });
	let cacheLogs = $derived($processLogs.rl_cache_rollouts || []);
	let trainLogs = $derived($processLogs.rl_train || []);

	const lossOptions = [
		{ value: 'nft', label: 'NFT — negative-aware (default)' },
		{ value: 'rwr', label: 'RWR — advantage-weighted regression' },
		{ value: 'dpo', label: 'DPO — group best vs worst' },
		{ value: 'ppo', label: 'PPO — clipped surrogate' }
	];
</script>

{#if !$projectLoaded}
	<div class="text-center py-16" style="color: var(--text-muted);">
		<p>No project loaded. Go to <a href="/" style="color: var(--accent);">Project</a> to create or load one.</p>
	</div>
{:else if !$advancedMode}
	<div class="space-y-4">
		<div class="p-5" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="text-[13px] font-semibold mb-1" style="color: var(--text-primary);">Advanced mode required</div>
			<div class="text-[12px]" style="color: var(--text-secondary);">Switch the left sidebar to `Advanced` to access RL post-training (reward-driven LoRA fine-tuning).</div>
		</div>
	</div>
{:else}
	<div class="space-y-5">
		<!-- Intro -->
		<div class="p-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			<div class="text-[13px] font-semibold mb-1" style="color: var(--text-primary);">Reinforcement-learning post-training (RL LoRA)</div>
			<p class="text-[12px] leading-relaxed" style="color: var(--text-secondary);">
				Refines an already-trained LoRA against a scalar <strong>reward</strong> instead of a paired dataset:
				sample K rollouts per prompt, score them, compute group-relative (GRPO) advantages, and update the LoRA
				with a negative-aware / advantage-weighted rule. Run <strong>Phase A</strong> (generate + score + cache rollouts),
				then <strong>Phase B</strong> (replay the cache and update the LoRA); warm-start the next round from the new LoRA.
			</p>
			<p class="text-[11px] leading-relaxed mt-2" style="color: var(--text-muted);">
				Model, LoRA (incl. the warm-start <code>--network_weights</code>), precision, memory and optimizer settings are
				inherited from the <a href="/training" style="color: var(--accent);">Training</a> tab — set those there first.
				Progress streams in the consoles below and on the <a href="/training/dashboard" style="color: var(--accent);">Monitor</a> tab.
				See the <em>RL post-training</em> section of the docs for prompt-count / group-size guidance and what to monitor.
			</p>
		</div>

		<!-- Phase A: rollout caching -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M9 3v4M15 3v4M4 11h16"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Phase A — Rollout caching</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Generate K rollouts per prompt, score with the reward zoo, write a disk cache</div>
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					This is the heavy, VRAM-hungry phase (the DiT plus, transiently, the text encoder and reward models). It writes
					clean rollouts + advantages to <code>--rl_rollout_cache</code> for Phase B to replay.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Prompts &amp; cache</div>
					<div class="space-y-2">
						<PathInput label="Prompts File" fieldPath="rl.rl_prompts" value={$projectConfig?.rl?.rl_prompts || ''} oninput={(e) => update('rl_prompts', e.target.value)} showFiles placeholder="rl_prompts.txt" tooltip="Text file of prompts (one per line). Diversity matters more than count; 8 is a smoke test, 50+ is a real run." />
						<PathInput label="Rollout Cache (output)" fieldPath="rl.rl_rollout_cache" value={$projectConfig?.rl?.rl_rollout_cache || ''} oninput={(e) => update('rl_rollout_cache', e.target.value)} tooltip="Directory the rollouts + scores + advantages are written to. Phase B reads this same path." />
						<PathInput label="Save `old` Snapshot (optional)" fieldPath="rl.rl_save_old_lora" value={$projectConfig?.rl?.rl_save_old_lora || ''} oninput={(e) => update('rl_save_old_lora', e.target.value)} showFiles placeholder="old_snapshot_r0.safetensors" tooltip="Writes the fp32 `old` snapshot that generated this cache. Load it as Network Weights in Phase B so `default` starts equal to `old` (the snapshot-hash invariant)." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Reward &amp; group</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Group Size (K)" type="number" fieldPath="rl.rl_group_size" value={$projectConfig?.rl?.rl_group_size ?? 8} oninput={(e) => update('rl_group_size', Number(e.target.value))} min={2} tooltip="K rollouts per prompt = GRPO group size. K>=4, 8-16 ideal; too small gives zero-variance groups." />
						<FormField label="Reward Function" fieldPath="rl.reward_fn" value={$projectConfig?.rl?.reward_fn || 'iqa_quality:1.0,anti_noise:0.1'} oninput={(e) => update('reward_fn', e.target.value)} placeholder="iqa_quality:1.0,anti_noise:0.1,videoreward:0.15" tooltip="Comma-separated name:weight. A bare name defaults to weight 1.0." />
						<FormField label="Reward Args" fieldPath="rl.reward_args" value={$projectConfig?.rl?.reward_args || ''} oninput={(e) => update('reward_args', e.target.value)} placeholder="checkpoint_path=/models/hpsv3" tooltip="key=value entries forwarded to every selected reward's setup()." />
					</div>
					<div class="mt-2">
						<FormField label="Reward Plugins" fieldPath="rl.reward_plugins" value={$projectConfig?.rl?.reward_plugins || ''} oninput={(e) => update('reward_plugins', e.target.value)} placeholder="path/to/my_reward.py" tooltip="Whitespace-separated custom-reward .py files (--reward_plugins); imported before the reward spec is parsed, so their names are usable in Reward Function." />
					</div>
				</div>

				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Rollout sampling</div>
					<div class="grid grid-cols-5 gap-2">
						<FormField label="Width" type="number" fieldPath="rl.sample_width" value={$projectConfig?.rl?.sample_width ?? 768} oninput={(e) => update('sample_width', Number(e.target.value))} min={16} step="16" tooltip="Rollout video width (default 768)." />
						<FormField label="Height" type="number" fieldPath="rl.sample_height" value={$projectConfig?.rl?.sample_height ?? 512} oninput={(e) => update('sample_height', Number(e.target.value))} min={16} step="16" tooltip="Rollout video height (default 512)." />
						<FormField label="Frames" type="number" fieldPath="rl.sample_frames" value={$projectConfig?.rl?.sample_frames ?? 49} oninput={(e) => update('sample_frames', Number(e.target.value))} min={1} tooltip="Rollout frame count (default 49)." />
						<FormField label="Steps" type="number" fieldPath="rl.sample_steps" value={$projectConfig?.rl?.sample_steps ?? 20} oninput={(e) => update('sample_steps', Number(e.target.value))} min={1} tooltip="Denoise steps per rollout (default 20). Fewer = faster generation." />
						<FormField label="CFG" type="number" fieldPath="rl.sample_cfg" value={$projectConfig?.rl?.sample_cfg ?? 1.0} oninput={(e) => update('sample_cfg', Number(e.target.value))} step="0.1" min={1} tooltip="Guidance scale for rollout generation (default 1.0)." />
					</div>
				</div>

				<FormField fieldPath="rl.extra_args" label="Extra Args" value={$projectConfig?.rl?.extra_args || ''} oninput={(e) => update('extra_args', e.target.value)} placeholder="--some_flag value" tooltip="Additional CLI flags appended to the Phase A command (shared with Phase B)." />

				<div class="mb-2">
					<ProcessControls processType="rl_cache_rollouts" status={cacheStatus} onStart={() => startProcess('rl_cache_rollouts')} onStop={() => stopProcess('rl_cache_rollouts')} />
				</div>
				<ProcessConsole lines={cacheLogs} processType="rl_cache_rollouts" />
				<CommandPanel processType="rl_cache_rollouts" defaultFilename="rl_cache_rollouts.bat" />
			</div>
		</div>

		<!-- Phase B: training -->
		<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
			<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), var(--secondary, var(--accent)), transparent); opacity: 0.5;"></div>

			<div class="p-5 pb-0">
				<div class="flex items-center gap-3 mb-2">
					<div class="w-8 h-8 flex items-center justify-center flex-shrink-0" style="background: var(--accent-muted); border-radius: var(--radius-sm);">
						<svg class="w-4 h-4" style="color: var(--accent);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
					</div>
					<div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">Phase B — Policy update</div>
						<div class="text-[11px]" style="color: var(--text-muted);">Replay the cache and run the three-forward update with the selected rule</div>
					</div>
				</div>
				<p class="text-[12px] leading-relaxed mb-3" style="color: var(--text-secondary);">
					The cheap phase (~8 GB at fp8 + block-swap + gradient checkpointing). Load the Phase A <code>old</code> snapshot as
					Network Weights on the Training tab so <code>default</code> starts equal to <code>old</code>.
				</p>
			</div>

			<div class="p-5 pt-0 space-y-3">
				<!-- Rollout source -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="flex items-center justify-between mb-2">
						<span class="text-[12px] font-semibold" style="color: var(--text-primary);">Rollout source</span>
						<FormToggle label="Online (generate inline)" fieldPath="rl.online" checked={online} onchange={(e) => update('online', e.target.checked)} tooltip="Off (recommended): replay a Phase A cache. On: generate rollouts inline each step (experimental, not VRAM-flat)." />
					</div>
					{#if !online}
						<PathInput label="Rollout Cache (from Phase A)" fieldPath="rl.rl_rollout_cache" value={$projectConfig?.rl?.rl_rollout_cache || ''} oninput={(e) => update('rl_rollout_cache', e.target.value)} tooltip="The directory written by Phase A. Phase B aborts if its `old` snapshot hash does not match the cache." />
					{:else}
						<div class="space-y-2">
							<PathInput label="Prompts File" fieldPath="rl.rl_prompts" value={$projectConfig?.rl?.rl_prompts || ''} oninput={(e) => update('rl_prompts', e.target.value)} showFiles placeholder="rl_prompts.txt" tooltip="Prompts to generate rollouts from inline (online mode)." />
							<div class="grid grid-cols-3 gap-2">
								<FormField label="Group Size (K)" type="number" fieldPath="rl.rl_group_size" value={$projectConfig?.rl?.rl_group_size ?? 8} oninput={(e) => update('rl_group_size', Number(e.target.value))} min={2} tooltip="K rollouts per prompt." />
								<FormField label="Reward Function" fieldPath="rl.reward_fn" value={$projectConfig?.rl?.reward_fn || 'iqa_quality:1.0,anti_noise:0.1'} oninput={(e) => update('reward_fn', e.target.value)} placeholder="iqa_quality:1.0,anti_noise:0.1" tooltip="Comma-separated name:weight." />
								<FormField label="Reward Args" fieldPath="rl.reward_args" value={$projectConfig?.rl?.reward_args || ''} oninput={(e) => update('reward_args', e.target.value)} placeholder="key=value" tooltip="Forwarded to each reward's setup()." />
							</div>
							<PathInput label="Dump Cache (optional)" fieldPath="rl.rl_dump_cache" value={$projectConfig?.rl?.rl_dump_cache || ''} oninput={(e) => update('rl_dump_cache', e.target.value)} tooltip="Also write the inline rollouts to disk (for the online==offline equivalence harness)." />
						</div>
					{/if}
				</div>

				<!-- Update rule -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Update rule</div>
					<FormSelect fieldPath="rl.rl_loss" value={rlLoss} onchange={(e) => update('rl_loss', e.target.value)} options={lossOptions} tooltip="All rules consume the same cached rollouts; only the final loss differs. NFT is the default." />
					{#if rlLoss === 'rwr'}
						<div class="grid grid-cols-3 gap-2 mt-2">
							<FormField label="RWR Temperature" type="number" fieldPath="rl.rwr_temperature" value={$projectConfig?.rl?.rwr_temperature ?? 1.0} oninput={(e) => update('rwr_temperature', Number(e.target.value))} step="0.1" min={0.01} tooltip="Softmax temperature on advantages (default 1.0). Lower = sharper weighting toward the best samples." />
						</div>
					{:else if rlLoss === 'dpo'}
						<div class="grid grid-cols-3 gap-2 mt-2">
							<FormField label="DPO Beta" type="number" fieldPath="rl.dpo_beta" value={$projectConfig?.rl?.dpo_beta ?? 5.0} oninput={(e) => update('dpo_beta', Number(e.target.value))} step="0.5" min={0.01} tooltip="Preference sharpness (default 5.0). Higher = stronger winner-over-loser preference." />
						</div>
					{:else if rlLoss === 'ppo'}
						<div class="grid grid-cols-3 gap-2 mt-2">
							<FormField label="PPO Clip Eps" type="number" fieldPath="rl.ppo_clip_eps" value={$projectConfig?.rl?.ppo_clip_eps ?? 0.2} oninput={(e) => update('ppo_clip_eps', Number(e.target.value))} step="0.05" min={0.01} tooltip="Importance-ratio clip epsilon (default 0.2)." />
							<FormField label="SDE eta" type="number" fieldPath="rl.rl_sde_eta" value={$projectConfig?.rl?.rl_sde_eta ?? 1.0} oninput={(e) => update('rl_sde_eta', Number(e.target.value))} step="0.1" min={0} max={1} tooltip="Per-step SDE noise level in [0,1] (1 = fully stochastic); must match Phase A. PPO is trajectory-faithful DDPO and needs the SDE sampler enabled in Phase A so the trajectory is cached." />
						</div>
					{/if}
				</div>

				<!-- Loss coefficients -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loss coefficients</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="NFT Beta Mix" type="number" fieldPath="rl.nft_beta_mix" value={$projectConfig?.rl?.nft_beta_mix ?? 1.0} oninput={(e) => update('nft_beta_mix', Number(e.target.value))} step="0.1" min={0.01} disabled={rlLoss !== 'nft'} tooltip="NFT positive/negative target mix (default 1.0). Used by the NFT rule only." />
						<FormField label="KL Beta" type="number" fieldPath="rl.nft_kl_beta" value={$projectConfig?.rl?.nft_kl_beta ?? 0.0001} oninput={(e) => update('nft_kl_beta', Number(e.target.value))} step="0.0001" min={0} disabled={rlLoss === 'dpo'} tooltip="KL-to-reference coefficient (default 1e-4). Anchors the policy to the frozen base. DPO uses its own preference anchor." />
						<FormField label="Advantage Clip Max" type="number" fieldPath="rl.nft_adv_clip_max" value={$projectConfig?.rl?.nft_adv_clip_max ?? 5.0} oninput={(e) => update('nft_adv_clip_max', Number(e.target.value))} step="0.5" min={0.01} tooltip="Clip on the group-relative advantage magnitude (default 5.0)." />
					</div>
				</div>

				<!-- Loop schedule -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Loop schedule</div>
					<div class="grid grid-cols-3 gap-2">
						<FormField label="Max Steps" type="number" fieldPath="rl.rl_max_steps" value={$projectConfig?.rl?.rl_max_steps ?? 0} oninput={(e) => update('rl_max_steps', Number(e.target.value))} min={0} tooltip="0 = ~one pass over the cache (recommended, since `old` is frozen per round). Advance the policy by regenerating the cache, not by extra passes." />
						<FormField label="Timesteps / Sample" type="number" fieldPath="rl.rl_timesteps_per_sample" value={$projectConfig?.rl?.rl_timesteps_per_sample ?? 1} oninput={(e) => update('rl_timesteps_per_sample', Number(e.target.value))} min={1} tooltip="Random re-noise timesteps drawn per cached sample (default 1)." />
						<FormField label="Decay Type" type="number" fieldPath="rl.rl_decay_type" value={$projectConfig?.rl?.rl_decay_type ?? 1} oninput={(e) => update('rl_decay_type', Number(e.target.value))} min={0} tooltip="`old` EMA decay schedule selector (default 1). `old` is a frozen per-round snapshot." />
					</div>
					<div class="mt-2">
						<FormToggle label="Log Phase-B VRAM peak" fieldPath="rl.rl_log_vram" checked={$projectConfig?.rl?.rl_log_vram ?? false} onchange={(e) => update('rl_log_vram', e.target.checked)} tooltip="Log the isolated peak VRAM of the training step (resets the peak counter before the loop)." />
					</div>
				</div>

				<!-- Output -->
				<div class="p-3" style="background: var(--bg-elevated); border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
					<div class="text-[11px] font-semibold mb-2" style="color: var(--text-primary);">Output</div>
					<div class="grid grid-cols-2 gap-2">
						<FormField label="Output Name" fieldPath="rl.output_name" value={$projectConfig?.rl?.output_name || 'ltx2_rl_lora'} oninput={(e) => update('output_name', e.target.value)} placeholder="ltx2_rl_lora" tooltip="Output LoRA name. Output directory is inherited from the Training tab." />
						<FormField label="Accelerate Args" fieldPath="rl.accelerate_extra_args" value={$projectConfig?.rl?.accelerate_extra_args || ''} oninput={(e) => update('accelerate_extra_args', e.target.value)} placeholder="--num_processes 1" tooltip="Extra arguments passed to `accelerate launch` for the Phase B process." />
					</div>
				</div>

				<div class="mb-2">
					<ProcessControls processType="rl_train" status={trainStatus} onStart={() => startProcess('rl_train')} onStop={() => stopProcess('rl_train')} />
				</div>
				<ProcessConsole lines={trainLogs} processType="rl_train" />
				<CommandPanel processType="rl_train" defaultFilename="rl_train.bat" />
			</div>
		</div>
	</div>
{/if}
