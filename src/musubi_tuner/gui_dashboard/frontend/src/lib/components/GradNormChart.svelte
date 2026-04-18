<script>
	import { onMount } from 'svelte';
	import * as echarts from 'echarts';
	import { gradNormData } from '../stores/metrics.js';

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
			grid: { left: 60, right: 20, top: 30, bottom: 50 },
			tooltip: {
				trigger: 'axis',
				backgroundColor: '#1f2937',
				borderColor: '#374151',
				textStyle: { color: '#e5e7eb', fontSize: 12 }
			},
			legend: {
				top: 6,
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
				name: 'Norm',
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

		const hasVideo = data.some((r) => r.grad_norm_v !== null && !isNaN(r.grad_norm_v));
		const hasAudio = data.some((r) => r.grad_norm_a !== null && !isNaN(r.grad_norm_a));

		const series = [
			{
				name: 'Total',
				type: 'line',
				data: data
					.filter((r) => r.grad_norm !== null && !isNaN(r.grad_norm))
					.map((r) => [r.step, r.grad_norm]),
				lineStyle: { width: 2 },
				symbol: 'none',
				color: '#f97316'
			}
		];

		if (hasVideo) {
			series.push({
				name: 'Video',
				type: 'line',
				data: data
					.filter((r) => r.grad_norm_v !== null && !isNaN(r.grad_norm_v))
					.map((r) => [r.step, r.grad_norm_v]),
				lineStyle: { width: 1.5 },
				symbol: 'none',
				color: '#10b981'
			});
		}

		if (hasAudio) {
			series.push({
				name: 'Audio',
				type: 'line',
				data: data
					.filter((r) => r.grad_norm_a !== null && !isNaN(r.grad_norm_a))
					.map((r) => [r.step, r.grad_norm_a]),
				lineStyle: { width: 1.5 },
				symbol: 'none',
				color: '#f59e0b'
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

	$effect(() => { queueUpdate($gradNormData); });
</script>

<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
	<h3 class="text-sm font-medium text-gray-400 mb-2">Grad Norm</h3>
	<div bind:this={container} class="w-full h-[350px]"></div>
</div>
