import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	optimizeDeps: {
		exclude: ['@duckdb/duckdb-wasm']
	},
	build: {
		chunkSizeWarningLimit: 1200,
		rollupOptions: {
			output: {
				manualChunks(id) {
					if (id.includes('node_modules/echarts')) return 'vendor-echarts';
					if (id.includes('node_modules/@duckdb')) return 'vendor-duckdb';
				}
			}
		}
	},
	server: {
		host: '127.0.0.1',
		hmr: true,
		proxy: {
			'/api': 'http://127.0.0.1:7860',
			'/data': 'http://127.0.0.1:7860',
			'/sse': 'http://127.0.0.1:7860'
		}
	}
});
