<script>
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import { dataWaitData } from '../stores/metrics.js';

	let container;
	let chart;
	let pendingFrame = 0;
	let pendingData = [];

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

	function movingAvg(arr, window = 20) {
		const result = [];
		for (let i = 0; i < arr.length; i++) {
			const start = Math.max(0, i - window + 1);
			const slice = arr.slice(start, i + 1);
			result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
		}
		return result;
	}

	function baseOption() {
		return {
			animation: false,
			animationDuration: 0,
			animationDurationUpdate: 0,
			backgroundColor: 'transparent',
			grid: { left: 60, right: 20, top: 30, bottom: 50 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1f2937',
				borderColor: '#374151',
				textStyle: { color: '#e5e7eb', fontSize: 12 },
				formatter: (params) => {
					const p = params[0];
					return `Step ${p.value[0]}<br/>${p.value[1].toFixed(2)}s wait`;
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
				name: 'Wait (s)',
				nameTextStyle: { color: '#6b7280' },
				axisLine: { lineStyle: { color: '#374151' } },
				axisLabel: { color: '#6b7280' },
				splitLine: { lineStyle: { color: '#1f2937' } }
			},
			dataZoom: [{ type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false }],
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
		const wait = data.map((r) => r.data_wait_time ?? 0);
		const smoothed = movingAvg(wait);

		chart.setOption({
			series: [
				{
					name: 'I/O wait (raw)',
					type: 'line',
					data: steps.map((s, i) => [s, wait[i]]),
					lineStyle: { width: 1, opacity: 0.3 },
					symbol: 'none',
					color: '#22c55e'
				},
				{
					name: 'I/O wait (avg)',
					type: 'line',
					data: steps.map((s, i) => [s, smoothed[i]]),
					lineStyle: { width: 1.5 },
					symbol: 'none',
					color: '#22c55e'
				}
			]
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

	$effect(() => { queueUpdate($dataWaitData); });
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<h3 class="text-sm font-medium text-gray-400 mb-2">Data I/O Wait</h3>
	<div bind:this={container} class="w-full h-[200px]"></div>
</div>
