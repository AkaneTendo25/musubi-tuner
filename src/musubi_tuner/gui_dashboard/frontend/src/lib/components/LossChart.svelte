	<script>
	import { onMount } from 'svelte';
		import * as echarts from 'echarts';
		import { lossData, hasAudioLoss, validationLossData } from '../stores/metrics.js';

		let container;
		let chart;
		let pendingFrame = 0;
		let pendingData = [];

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

		const series = [
			{
				name: 'Loss (raw)',
				type: 'line',
				data: lttbDownsample(steps.map((s, i) => [s, rawLoss[i]]), 2000),
				lineStyle: { width: 1.75 },
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
			series.push({
				name: 'Video Loss',
				type: 'line',
				data: lttbDownsample(
					steps.map((s, i) => [s, data[i].loss_v !== null && !isNaN(data[i].loss_v) ? data[i].loss_v : null]),
					2000
				),
				lineStyle: { width: 1.5 },
				connectNulls: false,
				symbol: 'none',
				color: '#10b981'
			});
		}

		// Audio loss
		if ($hasAudioLoss) {
			series.push({
				name: 'Audio Loss',
				type: 'line',
				data: lttbDownsample(
					steps.map((s, i) => [s, data[i].loss_a !== null && !isNaN(data[i].loss_a) ? data[i].loss_a : null]),
					2000
				),
				lineStyle: { width: 1.5 },
				connectNulls: false,
				symbol: 'none',
				color: '#f59e0b'
			});
		}

		if ($validationLossData.length) {
			series.push({
				name: 'Validation Loss',
				type: 'line',
				data: $validationLossData.map((r) => [r.step, r.val_loss]),
				lineStyle: { width: 1.5, type: 'dashed' },
				symbol: 'circle',
				symbolSize: 6,
				connectNulls: false,
				color: '#f97316'
			});
		}

		chart.setOption({ series }, { notMerge: false, lazyUpdate: true, replaceMerge: ['series'] });
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
		const ro = new ResizeObserver(() => chart?.resize());
		ro.observe(container);
		return () => {
			if (pendingFrame) cancelAnimationFrame(pendingFrame);
			ro.disconnect();
			chart?.dispose();
		};
	});

	$effect(() => {
		$hasAudioLoss;
		$validationLossData;
		queueUpdate($lossData);
	});
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<div class="flex items-center justify-between mb-2">
		<h3 class="text-sm font-medium text-gray-400">Loss</h3>
	</div>
	<div bind:this={container} class="w-full h-[350px]"></div>
</div>
