<script>
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import { lrData } from '../stores/metrics.js';

	let container;
	let chart;

	function updateChart(data) {
		if (!chart || !data.length) return;

		chart.setOption({
			animation: false,
			backgroundColor: 'transparent',
			grid: { left: 70, right: 20, top: 30, bottom: 50 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1f2937',
				borderColor: '#374151',
				textStyle: { color: '#e5e7eb', fontSize: 12 },
				formatter: (params) => {
					const p = params[0];
					return `Step ${p.value[0]}<br/>LR: ${p.value[1].toExponential(4)}`;
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
				type: 'log',
				name: 'Learning Rate',
				nameTextStyle: { color: '#6b7280' },
				axisLine: { lineStyle: { color: '#374151' } },
				axisLabel: {
					color: '#6b7280',
					formatter: (v) => v.toExponential(0)
				},
				splitLine: { lineStyle: { color: '#1f2937' } }
			},
			dataZoom: [{ type: 'inside', xAxisIndex: 0 }],
			series: [
				{
					type: 'line',
					data: data.map((r) => [r.step, r.lr]),
					lineStyle: { width: 1.5 },
					symbol: 'none',
					color: '#14b8a6'
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

	$effect(() => { updateChart($lrData); });
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<h3 class="text-sm font-medium text-gray-400 mb-2">Learning Rate</h3>
	<div bind:this={container} class="w-full h-[200px]"></div>
</div>
