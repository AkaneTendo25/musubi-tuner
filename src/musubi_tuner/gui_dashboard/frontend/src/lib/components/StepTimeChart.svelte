<script>
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import { stepTimeData } from '../stores/metrics.js';

	let container;
	let chart;

	function movingAvg(arr, window = 20) {
		const result = [];
		for (let i = 0; i < arr.length; i++) {
			const start = Math.max(0, i - window + 1);
			const slice = arr.slice(start, i + 1);
			result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
		}
		return result;
	}

	function updateChart(data) {
		if (!chart || !data.length) return;

		const steps = data.map((r) => r.step);
		const times = data.map((r) => r.step_time);
		const smoothed = movingAvg(times);

		chart.setOption({
			animation: false,
			backgroundColor: 'transparent',
			grid: { left: 60, right: 20, top: 30, bottom: 50 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1f2937',
				borderColor: '#374151',
				textStyle: { color: '#e5e7eb', fontSize: 12 },
				formatter: (params) => {
					const p = params[0];
					return `Step ${p.value[0]}<br/>${p.value[1].toFixed(2)}s/step`;
				}
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
				name: 'Time (s)',
				nameTextStyle: { color: '#6b7280' },
				axisLine: { lineStyle: { color: '#374151' } },
				axisLabel: { color: '#6b7280' },
				splitLine: { lineStyle: { color: '#1f2937' } }
			},
			dataZoom: [{ type: 'inside', xAxisIndex: 0 }],
			series: [
				{
					name: 'Step time (raw)',
					type: 'line',
					data: steps.map((s, i) => [s, times[i]]),
					lineStyle: { width: 1, opacity: 0.3 },
					symbol: 'none',
					color: '#f472b6'
				},
				{
					name: 'Step time (avg)',
					type: 'line',
					data: steps.map((s, i) => [s, smoothed[i]]),
					lineStyle: { width: 1.5 },
					symbol: 'none',
					color: '#f472b6'
				}
			]
		}, true);
	}

	onMount(() => {
		chart = echarts.init(container, null, { renderer: 'canvas' });
		const ro = new ResizeObserver(() => chart?.resize());
		ro.observe(container);
		return () => { ro.disconnect(); chart?.dispose(); };
	});

	$effect(() => { updateChart($stepTimeData); });
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<h3 class="text-sm font-medium text-gray-400 mb-2">Step Time</h3>
	<div bind:this={container} class="w-full h-[200px]"></div>
</div>
