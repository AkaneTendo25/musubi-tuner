<script>
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import { validationLossData } from '../stores/metrics.js';

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

	function baseOption() {
		return {
			animation: false,
			animationDuration: 0,
			animationDurationUpdate: 0,
			backgroundColor: 'transparent',
			grid: { left: 60, right: 20, top: 30, bottom: 60 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1f2937',
				borderColor: '#374151',
				textStyle: { color: '#e5e7eb', fontSize: 12 },
				formatter: (params) => {
					const p = params[0];
					if (!p) return '';
					return `Step ${p.value[0]}<br/>Val Loss: ${Number(p.value[1]).toFixed(4)}`;
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
				name: 'Val Loss',
				nameTextStyle: { color: '#6b7280' },
				axisLine: { lineStyle: { color: '#374151' } },
				axisLabel: { color: '#6b7280' },
				splitLine: { lineStyle: { color: '#1f2937' } }
			},
			dataZoom: [
				{ type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false },
				{
					type: 'slider', xAxisIndex: 0, height: 20, bottom: 8,
					borderColor: '#374151', fillerColor: 'rgba(168,85,247,0.10)',
					handleStyle: { color: '#a855f7' },
					textStyle: { color: '#6b7280' }
				}
			],
			graphic: [
				{
					id: 'empty',
					type: 'text',
					left: 'center',
					top: 'middle',
					silent: true,
					style: {
						text: 'Validation loss will appear here when validation runs',
						fill: '#6b7280',
						fontSize: 12
					}
				}
			],
			series: []
		};
	}

	function renderChart(data) {
		if (!chart) return;
		if (!data.length) {
			chart.setOption(
				{
					graphic: [{ id: 'empty', invisible: false }],
					series: []
				},
				{ notMerge: false, lazyUpdate: true, replaceMerge: ['series', 'graphic'] }
			);
			return;
		}

		chart.setOption(
			{
				graphic: [{ id: 'empty', invisible: true }],
				series: [
					{
						name: 'Validation Loss',
						type: 'line',
						data: data.map((r) => [r.step, r.val_loss]),
						lineStyle: { width: 2 },
						symbol: 'circle',
						symbolSize: 6,
						color: '#a855f7'
					}
				]
			},
			{ notMerge: false, lazyUpdate: true, replaceMerge: ['series', 'graphic'] }
		);
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
		queueUpdate($validationLossData);
	});
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<h3 class="text-sm font-medium text-gray-400 mb-2">Validation Loss</h3>
	<div bind:this={container} class="w-full h-[350px]"></div>
</div>
