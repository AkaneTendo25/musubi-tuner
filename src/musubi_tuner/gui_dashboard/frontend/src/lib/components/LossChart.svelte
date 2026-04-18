<script>
	import { onMount, onDestroy } from 'svelte';
	import * as echarts from 'echarts';
	import { lossData, hasAudioLoss } from '../stores/metrics.js';
	import SmoothnessSlider from './SmoothnessSlider.svelte';

	let container;
	let chart;
	let pendingFrame = 0;
	let pendingData = [];
	let smoothing = $state(0);
	let logScale = $state(false);

	function getScrollTarget(node) {
		let current = node?.parentElement;
		while (current) {
			const style = getComputedStyle(current);
			if (/(auto|scroll)/.test(style.overflowY) && current.scrollHeight > current.clientHeight) {
				return current;
			}
			current = current.parentElement;
		}
		return document.scrollingElement || document.documentElement;
	}

	function handleWheel(event) {
		if (event.ctrlKey) return;
		event.preventDefault();
		event.stopPropagation();
		const target = getScrollTarget(container);
		target?.scrollBy?.({ top: event.deltaY, left: event.deltaX, behavior: 'auto' });
	}

	function ema(data, alpha) {
		if (!data.length) return [];
		const result = [data[0]];
		for (let i = 1; i < data.length; i++) {
			result.push(alpha * result[i - 1] + (1 - alpha) * data[i]);
		}
		return result;
	}

	function lttbDownsample(data, threshold) {
		if (data.length <= threshold) return data;
		const sampled = [data[0]];
		const every = (data.length - 2) / (threshold - 2);
		let a = 0;
		for (let i = 0; i < threshold - 2; i++) {
			let avgX = 0, avgY = 0, count = 0;
			const rangeStart = Math.floor((i + 1) * every) + 1;
			const rangeEnd = Math.min(Math.floor((i + 2) * every) + 1, data.length);
			for (let j = rangeStart; j < rangeEnd; j++) {
				avgX += data[j][0]; avgY += data[j][1]; count++;
			}
			avgX /= count; avgY /= count;

			const bucketStart = Math.floor(i * every) + 1;
			const bucketEnd = Math.min(Math.floor((i + 1) * every) + 1, data.length);
			let maxArea = -1, maxIdx = bucketStart;
			for (let j = bucketStart; j < bucketEnd; j++) {
				const area = Math.abs(
					(data[a][0] - avgX) * (data[j][1] - data[a][1]) -
					(data[a][0] - data[j][0]) * (avgY - data[a][1])
				);
				if (area > maxArea) { maxArea = area; maxIdx = j; }
			}
			sampled.push(data[maxIdx]);
			a = maxIdx;
		}
		sampled.push(data[data.length - 1]);
		return sampled;
	}

	function baseOption() {
		return {
			animation: false,
			animationDuration: 0,
			animationDurationUpdate: 0,
			backgroundColor: 'transparent',
			grid: { left: 60, right: 20, top: 40, bottom: 60 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1f2937',
				borderColor: '#374151',
				textStyle: { color: '#e5e7eb', fontSize: 12 },
				axisPointer: { type: 'cross', lineStyle: { color: '#4b5563' } }
			},
			legend: {
				top: 8,
				textStyle: { color: '#9ca3af', fontSize: 11 },
				icon: 'roundRect',
				itemWidth: 14,
				itemHeight: 3
			},
			xAxis: {
				type: 'value',
				name: 'Step',
				nameTextStyle: { color: '#6b7280' },
				axisLine: { lineStyle: { color: '#374151' } },
				axisLabel: { color: '#6b7280' },
				splitLine: { lineStyle: { color: '#1f2937' } }
			},
			yAxis: {
				type: 'value',
				name: 'Loss',
				nameTextStyle: { color: '#6b7280' },
				axisLine: { lineStyle: { color: '#374151' } },
				axisLabel: { color: '#6b7280' },
				splitLine: { lineStyle: { color: '#1f2937' } }
			},
			dataZoom: [
				{ type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false },
				{
					type: 'slider', xAxisIndex: 0, height: 20, bottom: 8,
					borderColor: '#374151', fillerColor: 'rgba(59,130,246,0.1)',
					handleStyle: { color: '#3b82f6' },
					textStyle: { color: '#6b7280' }
				}
			],
			series: []
		};
	}

	function renderChart(data) {
		if (!chart) return;
		if (!data.length) {
			chart.setOption({ series: [] }, { notMerge: false, lazyUpdate: true, replaceMerge: ['series'] });
			return;
		}

		const steps = data.map((r) => r.step);
		const rawLoss = data.map((r) => r.loss);
		const smoothed = ema(rawLoss, smoothing);

		const series = [
			{
				name: 'Loss (raw)',
				type: 'line',
				data: lttbDownsample(steps.map((s, i) => [s, rawLoss[i]]), 2000),
				lineStyle: { width: 1, opacity: 0.3 },
				symbol: 'none',
				color: '#3b82f6'
			},
			{
				name: 'Loss (smooth)',
				type: 'line',
				data: lttbDownsample(steps.map((s, i) => [s, smoothed[i]]), 2000),
				lineStyle: { width: 2 },
				symbol: 'none',
				color: '#3b82f6'
			},
			{
				name: 'Avg Loss',
				type: 'line',
				data: lttbDownsample(steps.map((s, i) => [s, data[i].avr_loss]), 2000),
				lineStyle: { width: 1.5, type: 'dashed' },
				symbol: 'none',
				color: '#8b5cf6'
			}
		];

		// Video loss (if separate)
		const hasVideo = data.some((r) => r.loss_v !== null && !isNaN(r.loss_v));
		if (hasVideo) {
			const vLoss = data.map((r) => (r.loss_v !== null && !isNaN(r.loss_v) ? r.loss_v : null));
			const vSmoothed = ema(vLoss.map(v => v ?? 0), smoothing);
			series.push({
				name: 'Video Loss',
				type: 'line',
				data: lttbDownsample(steps.map((s, i) => [s, vSmoothed[i]]), 2000),
				lineStyle: { width: 1.5 },
				symbol: 'none',
				color: '#10b981'
			});
		}

		// Audio loss
		if ($hasAudioLoss) {
			const aLoss = data.map((r) => (r.loss_a !== null && !isNaN(r.loss_a) ? r.loss_a : null));
			const aSmoothed = ema(aLoss.map(v => v ?? 0), smoothing);
			series.push({
				name: 'Audio Loss',
				type: 'line',
				data: lttbDownsample(steps.map((s, i) => [s, aSmoothed[i]]), 2000),
				lineStyle: { width: 1.5 },
				symbol: 'none',
				color: '#f59e0b'
			});
		}

		chart.setOption({
			yAxis: { type: logScale ? 'log' : 'value' },
			series
		}, { notMerge: false, lazyUpdate: true, replaceMerge: ['series'] });
	}

	function queueUpdate(data) {
		pendingData = data;
		if (!chart || pendingFrame) return;
		pendingFrame = requestAnimationFrame(() => {
			pendingFrame = 0;
			renderChart(pendingData);
		});
	}

	onMount(() => {
		chart = echarts.init(container, null, { renderer: 'canvas' });
		chart.setOption(baseOption(), { notMerge: true });
		container?.addEventListener('wheel', handleWheel, { passive: false, capture: true });
		const ro = new ResizeObserver(() => chart?.resize());
		ro.observe(container);
		return () => {
			if (pendingFrame) cancelAnimationFrame(pendingFrame);
			container?.removeEventListener('wheel', handleWheel, { capture: true });
			ro.disconnect();
			chart?.dispose();
		};
	});

	$effect(() => {
		smoothing;
		logScale;
		$hasAudioLoss;
		queueUpdate($lossData);
	});
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<div class="flex items-center justify-between mb-2">
		<h3 class="text-sm font-medium text-gray-400">Loss</h3>
		<div class="flex items-center gap-4">
			<SmoothnessSlider bind:value={smoothing} />
			<label class="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
				<input type="checkbox" bind:checked={logScale} class="accent-blue-500" />
				Log scale
			</label>
		</div>
	</div>
	<div bind:this={container} class="w-full h-[350px]"></div>
</div>
