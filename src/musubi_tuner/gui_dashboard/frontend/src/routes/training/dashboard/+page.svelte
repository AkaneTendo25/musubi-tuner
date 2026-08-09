<script>
	import StatusBar from '$lib/components/StatusBar.svelte';
	import LossChart from '$lib/components/LossChart.svelte';
	import LRChart from '$lib/components/LRChart.svelte';
	import StepTimeChart from '$lib/components/StepTimeChart.svelte';
	import GradNormChart from '$lib/components/GradNormChart.svelte';
	import DataWaitChart from '$lib/components/DataWaitChart.svelte';
	import ComputeTimeChart from '$lib/components/ComputeTimeChart.svelte';
	import CircularMetricGauge from '$lib/components/CircularMetricGauge.svelte';
	import SampleGallery from '$lib/components/SampleGallery.svelte';
	import ProcessConsole from '$lib/components/ProcessConsole.svelte';
	import ProcessControls from '$lib/components/ProcessControls.svelte';
	import { projectConfig } from '$lib/stores/project.js';
	import { processStatuses, processLogs, startProcess, stopProcess, preloadLogsIfActive, clearProcessLogs, refreshStatuses, fetchLogs, startLogPolling } from '$lib/stores/processes.js';
	import { clearMetrics, lossData, gradNormData, lrData, stepTimeData, dataWaitData } from '$lib/stores/metrics.js';
	import { status, clearStatus } from '$lib/stores/status.js';
	import { onMount } from 'svelte';

	let projectName = $derived($projectConfig?.name || 'Untitled');

	let rawTrainingStatus = $derived($processStatuses.training || { state: 'idle', exit_code: null });
	let rawFullFinetuneStatus = $derived($processStatuses.full_finetune || { state: 'idle', exit_code: null });
	let rawSliderStatus = $derived($processStatuses.slider_training || { state: 'idle', exit_code: null });
	let activeProcessType = $derived.by(() => {
		if (rawFullFinetuneStatus.state === 'running' || rawFullFinetuneStatus.state === 'stopping') return 'full_finetune';
		if (rawSliderStatus.state === 'running' || rawSliderStatus.state === 'stopping') return 'slider_training';
		if (rawTrainingStatus.state === 'running' || rawTrainingStatus.state === 'stopping') return 'training';
		if (rawFullFinetuneStatus.state === 'finished' || rawFullFinetuneStatus.state === 'error') return 'full_finetune';
		if (rawSliderStatus.state === 'finished' || rawSliderStatus.state === 'error') return 'slider_training';
		return 'training';
	});
	// Slider training inherits output_dir and most settings from the training config section.
	let t = $derived(activeProcessType === 'full_finetune' ? ($projectConfig?.full_finetune || {}) : ($projectConfig?.training || {}));
	let trainingLive = $derived.by(() => {
		if (activeProcessType === 'full_finetune') return rawFullFinetuneStatus.state === 'running' || rawFullFinetuneStatus.state === 'stopping';
		if (activeProcessType === 'slider_training') return rawSliderStatus.state === 'running' || rawSliderStatus.state === 'stopping';
		return rawTrainingStatus.state === 'running' || rawTrainingStatus.state === 'stopping';
	});
	let trainingStatus = $derived.by(() => {
		if (activeProcessType === 'full_finetune') return rawFullFinetuneStatus;
		if (activeProcessType === 'slider_training') return rawSliderStatus;
		return rawTrainingStatus;
	});
	let trainingLogs = $derived($processLogs[activeProcessType] || []);
	let trainingActive = $derived(trainingLive);
	let showDashboardData = $derived(trainingLive);
	let trainingConsoleEmptyMessage = $derived(
		trainingLive ? 'Training process launched. Waiting for console output...' : 'No output yet'
	);
	let systemInfo = $state(null);
	let stats = $state(null);
	let peakVramMb = $state(0);
	let lastRunActive = $state(false);
	let statsTimer = null;
	let systemInfoLoading = false;
	let statsLoading = false;

	async function fetchSystemInfo() {
		if (systemInfoLoading) return;
		systemInfoLoading = true;
		try {
			const res = await fetch('/api/system/info', { cache: 'no-store' });
			if (res.ok) systemInfo = await res.json();
		} catch {
		} finally {
			systemInfoLoading = false;
		}
	}

	async function fetchStats() {
		if (statsLoading) return;
		statsLoading = true;
		try {
			const res = await fetch('/api/stats', { cache: 'no-store' });
			if (res.ok) stats = await res.json();
		} catch {
		} finally {
			statsLoading = false;
		}
	}

	onMount(() => {
		let disposed = false;
		const allProcessTypes = ['training', 'full_finetune', 'slider_training'];
		const init = async () => {
			const statuses = await refreshStatuses();
			if (disposed) return;
			const trainingState = statuses?.training?.state;
			const fullFinetuneState = statuses?.full_finetune?.state;
			const sliderState = statuses?.slider_training?.state;
			const anyActive = [trainingState, fullFinetuneState, sliderState].some(
				s => s === 'running' || s === 'stopping'
			);
			if (anyActive) {
				await preloadLogsIfActive(allProcessTypes);
			} else {
				clearMetrics();
				clearStatus();
				for (const pt of allProcessTypes) clearProcessLogs(pt);
			}

			await Promise.all([fetchSystemInfo(), fetchStats()]);
		};
		void init();

		const systemInterval = setInterval(fetchSystemInfo, 1000);
		const statsInterval = setInterval(fetchStats, 3000);
		const logInterval = startLogPolling(allProcessTypes, 1000);
		return () => {
			disposed = true;
			clearInterval(systemInterval);
			clearInterval(statsInterval);
			clearInterval(logInterval);
		};
	});

	$effect(() => {
		const cfg = $projectConfig;
		if (!cfg) {
			stats = null;
			return;
		}

		const snapshot = JSON.stringify({
			dataset: cfg.dataset,
			activeProcessType,
			training: {
				gradient_accumulation_steps: cfg.training?.gradient_accumulation_steps,
				save_every_n_steps: cfg.training?.save_every_n_steps,
				sample_every_n_steps: cfg.training?.sample_every_n_steps,
				sample_every_n_epochs: cfg.training?.sample_every_n_epochs,
				output_dir: cfg.training?.output_dir,
				output_name: cfg.training?.output_name,
				resume: cfg.training?.resume,
				autoresume: cfg.training?.autoresume,
				resume_from_huggingface: cfg.training?.resume_from_huggingface,
				max_train_steps: cfg.training?.max_train_steps
			},
			full_finetune: {
				gradient_accumulation_steps: cfg.full_finetune?.gradient_accumulation_steps,
				save_every_n_steps: cfg.full_finetune?.save_every_n_steps,
				sample_every_n_steps: cfg.full_finetune?.sample_every_n_steps,
				sample_every_n_epochs: cfg.full_finetune?.sample_every_n_epochs,
				output_dir: cfg.full_finetune?.output_dir,
				output_name: cfg.full_finetune?.output_name,
				resume: cfg.full_finetune?.resume,
				autoresume: cfg.full_finetune?.autoresume,
				resume_from_huggingface: cfg.full_finetune?.resume_from_huggingface,
				max_train_steps: cfg.full_finetune?.max_train_steps
			}
		});

		if (statsTimer) clearTimeout(statsTimer);
		statsTimer = setTimeout(fetchStats, 700);

		return () => {
			if (statsTimer) clearTimeout(statsTimer);
		};
	});

	let gpu = $derived(systemInfo?.gpus?.[0]);
	let runStats = $derived(stats?.training || null);

	$effect(() => {
		if (trainingLive && !lastRunActive) {
			peakVramMb = 0;
		}
		lastRunActive = trainingLive;

		const used = gpu?.vram_used_mb ?? 0;
		if (trainingLive && used > peakVramMb) {
			peakVramMb = used;
		}
	});

	async function handleStart() {
		await startProcess(activeProcessType);
		await fetchLogs(activeProcessType);
	}

	async function handleStop() {
		await stopProcess(activeProcessType);
	}

	function formatDuration(seconds) {
		if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return '--';
		const h = Math.floor(seconds / 3600);
		const m = Math.floor((seconds % 3600) / 60);
		const s = Math.floor(seconds % 60);
		if (h > 0) return `${h}h ${m}m`;
		if (m > 0) return `${m}m ${s}s`;
		return `${s}s`;
	}

	function formatStorageGb(value) {
		if (value == null || !Number.isFinite(value)) return '--';
		if (value >= 100) return `${Math.round(value)} GB`;
		if (value >= 10) return `${value.toFixed(1)} GB`;
		return `${value.toFixed(2)} GB`;
	}

	function clampPercent(value) {
		const n = Number(value);
		if (!Number.isFinite(n)) return 0;
		return Math.max(0, Math.min(100, n));
	}

	function ratioPercent(used, total) {
		const u = Number(used);
		const t = Number(total);
		if (!Number.isFinite(u) || !Number.isFinite(t) || t <= 0) return 0;
		return clampPercent((u / t) * 100);
	}

	function truncateMiddle(value, max = 34) {
		if (!value) return '--';
		if (value.length <= max) return value;
		const edge = Math.max(8, Math.floor((max - 1) / 2));
		return `${value.slice(0, edge)}…${value.slice(-edge)}`;
	}

	function resumeModeLabel(training) {
		if (!training) return 'Fresh';
		if (training.resume_from_huggingface) return 'HuggingFace';
		if (training.resume) return 'Manual';
		if (training.autoresume) return 'Auto';
		return 'Fresh';
	}

	function nextStepEventLabel(interval, step, speed, offLabel = 'Off') {
		if (!interval || interval <= 0) return offLabel;
		const current = Math.max(Number(step || 0), 0);
		const next = Math.ceil((current + 1) / interval) * interval;
		const remaining = Math.max(next - current, 0);
		if (remaining === 0) return 'Due now';
		if (speed && Number.isFinite(speed) && speed > 0) {
			return `${formatDuration(remaining / speed)} (${remaining} steps)`;
		}
		return `${remaining} steps`;
	}

	function nextSampleLabel(training, liveStatus, trainingStats) {
		const stepInterval = Number(training?.sample_every_n_steps || 0);
		if (stepInterval > 0) {
			return nextStepEventLabel(stepInterval, liveStatus?.step, liveStatus?.speed_steps_per_sec);
		}

		const epochInterval = Number(training?.sample_every_n_epochs || 0);
		if (epochInterval > 0) {
			const stepsPerEpoch = Number(trainingStats?.steps_per_epoch || 0);
			if (stepsPerEpoch > 0 && liveStatus?.step != null) {
				const completedEpochs = Math.floor(Number(liveStatus.step) / stepsPerEpoch);
				const nextEpoch = Math.ceil((completedEpochs + 1) / epochInterval) * epochInterval;
				const remainingSteps = Math.max(nextEpoch * stepsPerEpoch - Number(liveStatus.step), 0);
				if (remainingSteps === 0) return 'Due this epoch';
				if (liveStatus?.speed_steps_per_sec > 0) {
					return `${formatDuration(remainingSteps / liveStatus.speed_steps_per_sec)} (${remainingSteps} steps)`;
				}
				return `${remainingSteps} steps`;
			}
			return `Every ${epochInterval} epochs`;
		}

		return 'Off';
	}

	let effectiveOutputDir = $derived(t.output_dir || 'output');
	let effectiveOutputName = $derived(t.output_name || 'ltx2_lora');
	let effectiveBatchSize = $derived(runStats?.effective_batch_size ?? null);
	let stepsPerEpoch = $derived(runStats?.steps_per_epoch ?? null);
	let nextCheckpoint = $derived(nextStepEventLabel(Number(t.save_every_n_steps || 0), $status?.step, $status?.speed_steps_per_sec, 'Final only'));
	let nextSample = $derived(nextSampleLabel(t, $status, runStats));
	let resumeMode = $derived(resumeModeLabel(t));
	let gpuVramPercent = $derived(ratioPercent(gpu?.vram_used_mb, gpu?.vram_total_mb));
	let gpuUtilization = $derived(clampPercent(gpu?.utilization));
	let fanPercent = $derived(gpu?.fan_speed_percent == null ? null : clampPercent(gpu.fan_speed_percent));
	let ramPercent = $derived(clampPercent(systemInfo?.ram?.percent));
	let diskFreePercent = $derived(ratioPercent(systemInfo?.disk?.free_gb, systemInfo?.disk?.total_gb));
	let cpuUtilization = $derived(systemInfo?.cpu?.utilization == null ? null : clampPercent(systemInfo.cpu.utilization));
	let hasLossChartData = $derived(($lossData?.length ?? 0) > 0);
	let hasGradNormChartData = $derived(
		($gradNormData?.some((r) =>
			(r.grad_norm !== null && !isNaN(r.grad_norm)) ||
			(r.grad_norm_v !== null && !isNaN(r.grad_norm_v)) ||
			(r.grad_norm_a !== null && !isNaN(r.grad_norm_a))
		) ?? false)
	);
	let hasBottomCharts = $derived(
		($lrData?.length ?? 0) > 0 ||
		($stepTimeData?.length ?? 0) > 0 ||
		($dataWaitData?.length ?? 0) > 0
	);
</script>

<div class="space-y-4">
	<!-- Status + Hardware row -->
	<StatusBar active={trainingActive} />

	<div class="grid grid-cols-1 xl:grid-cols-3 gap-3">
		<div class="p-3.5 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
			<div class="flex items-center justify-between gap-3">
				<div>
					<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">GPU</div>
					<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">{gpu?.name || 'No GPU detected'}</div>
				</div>
				{#if gpu?.power_draw_w != null}
					<div class="text-right">
						<div class="text-[10px]" style="color: var(--text-muted);">Power</div>
						<div class="text-[12px] font-semibold tabular-nums" style="color: var(--warning);">{gpu.power_draw_w.toFixed(0)} W</div>
					</div>
				{/if}
			</div>

			{#if gpu}
				<div class="grid grid-cols-2 2xl:grid-cols-3 gap-3">
					<CircularMetricGauge
						label="VRAM"
						value={gpuVramPercent}
						display={`${(gpu.vram_used_mb / 1024).toFixed(1)}G`}
						sublabel={`/ ${(gpu.vram_total_mb / 1024).toFixed(0)}G`}
						color="var(--accent)"
						size={116}
					/>
					<CircularMetricGauge
						label="GPU Load"
						value={gpuUtilization}
						display={gpu.utilization == null ? '--' : `${gpuUtilization.toFixed(0)}%`}
						color="var(--info)"
						size={116}
						inactive={gpu.utilization == null}
					/>
					{#if fanPercent != null}
						<CircularMetricGauge
							label="Fan"
							value={fanPercent}
							display={`${fanPercent.toFixed(0)}%`}
							color="var(--warning)"
							size={116}
						/>
					{/if}
				</div>

				<div class="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
					<div class="flex items-center justify-between gap-3">
						<span style="color: var(--text-muted);">Peak VRAM</span>
						<span class="font-medium tabular-nums" style="color: var(--text-primary);">{peakVramMb > 0 ? `${(peakVramMb / 1024).toFixed(1)} GB` : '--'}</span>
					</div>
					<div class="flex items-baseline justify-between gap-3">
						<span style="color: var(--text-muted);">Temp</span>
						<span class="text-[16px] leading-none font-bold tabular-nums" style="color: {gpu.temperature > 80 ? 'var(--danger)' : gpu.temperature > 72 ? 'var(--warning)' : 'var(--text-primary)'};">{gpu.temperature != null ? `${gpu.temperature}°C` : '--'}</span>
					</div>
					<div class="flex items-center justify-between gap-3">
						<span style="color: var(--text-muted);">Clock</span>
						<span class="font-medium tabular-nums" style="color: var(--text-primary);">{gpu.graphics_clock_mhz != null ? `${gpu.graphics_clock_mhz} MHz` : '--'}</span>
					</div>
				</div>
			{/if}
		</div>

		<div class="p-3.5 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
			<div>
				<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Run</div>
				<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">{effectiveOutputName}</div>
			</div>

			<div class="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Project</span>
					<span class="font-medium" style="color: var(--text-primary);">{truncateMiddle(projectName, 22)}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Resume</span>
					<span class="font-medium" style="color: var(--text-primary);">{resumeMode}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Eff. Batch</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{effectiveBatchSize ?? '--'}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Grad Accum</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{t.gradient_accumulation_steps ?? 1}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Steps / Epoch</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{stepsPerEpoch ?? '--'}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Storage</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{formatStorageGb(runStats?.total_storage_gb)}</span>
				</div>
			</div>

			<div class="grid grid-cols-1 gap-2 text-[11px]">
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Next checkpoint</span>
					<span class="font-medium tabular-nums text-right" style="color: var(--text-primary);">{nextCheckpoint}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Next sample</span>
					<span class="font-medium tabular-nums text-right" style="color: var(--text-primary);">{nextSample}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Output dir</span>
					<span class="font-medium text-right" style="color: var(--text-primary);" title={effectiveOutputDir}>{truncateMiddle(effectiveOutputDir, 34)}</span>
				</div>
			</div>
		</div>

		<div class="p-3.5 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); box-shadow: var(--shadow-sm);">
			<div>
				<div class="text-[10px] font-medium uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">System</div>
				<div class="text-[13px] font-semibold mt-1" style="color: var(--text-primary);">Host health</div>
			</div>

			{#if systemInfo?.ram || systemInfo?.disk || cpuUtilization != null}
				<div class="grid grid-cols-2 2xl:grid-cols-3 gap-3">
					{#if systemInfo?.ram}
						<CircularMetricGauge
							label="RAM"
							value={ramPercent}
							display={`${systemInfo.ram.used_gb}G`}
							sublabel={`/ ${systemInfo.ram.total_gb}G`}
							color={ramPercent > 90 ? 'var(--danger)' : ramPercent > 75 ? 'var(--warning)' : 'var(--info)'}
							size={116}
						/>
					{/if}
					{#if systemInfo?.disk}
						<CircularMetricGauge
							label="Disk Free"
							value={diskFreePercent}
							display={`${systemInfo.disk.free_gb}G`}
							sublabel={`/ ${systemInfo.disk.total_gb}G`}
							color={diskFreePercent < 10 ? 'var(--danger)' : diskFreePercent < 20 ? 'var(--warning)' : 'var(--success)'}
							size={116}
						/>
					{/if}
					{#if cpuUtilization != null}
						<CircularMetricGauge
							label="CPU Load"
							value={cpuUtilization}
							display={`${cpuUtilization.toFixed(0)}%`}
							color={cpuUtilization > 90 ? 'var(--danger)' : cpuUtilization > 75 ? 'var(--warning)' : 'var(--info)'}
							size={116}
						/>
					{/if}
				</div>
			{/if}

			<div class="grid grid-cols-2 gap-x-4 gap-y-2 text-[11px]">
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Checkpoints</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{runStats?.total_checkpoints ?? '--'}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Est. duration</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{runStats?.estimated_time_hours ? formatDuration(runStats.estimated_time_hours * 3600) : '--'}</span>
				</div>
				<div class="flex items-center justify-between gap-3">
					<span style="color: var(--text-muted);">Cores</span>
					<span class="font-medium tabular-nums" style="color: var(--text-primary);">{systemInfo?.cpu?.cores ?? '--'}</span>
				</div>
			</div>
		</div>
	</div>

	<!-- Process Controls + Console — always visible -->
	<div class="space-y-3">
		<ProcessControls processType={activeProcessType} status={trainingStatus} onStart={handleStart} onStop={handleStop} />
		<ProcessConsole lines={trainingLogs} processType={activeProcessType} initiallyCollapsed={false} emptyMessage={trainingConsoleEmptyMessage} />
	</div>

	{#if showDashboardData}
		{#if hasLossChartData || hasGradNormChartData}
			<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
				{#if hasLossChartData}
					<LossChart />
				{/if}
				{#if hasGradNormChartData}
					<GradNormChart />
				{/if}
			</div>
		{:else}
			<div class="p-4">
				<h3 class="text-sm font-medium mb-3" style="color: var(--text-secondary);">Charts</h3>
				<p class="text-sm" style="color: var(--text-muted);">No training metrics yet</p>
			</div>
		{/if}

		{#if hasBottomCharts}
			<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
				<LRChart />
				<StepTimeChart />
				<DataWaitChart />
				<ComputeTimeChart />
			</div>
		{:else}
			<div class="p-4">
				<h3 class="text-sm font-medium mb-3" style="color: var(--text-secondary);">Detailed Charts</h3>
				<p class="text-sm" style="color: var(--text-muted);">No detailed metrics yet</p>
			</div>
		{/if}

		<SampleGallery />
	{/if}
</div>
