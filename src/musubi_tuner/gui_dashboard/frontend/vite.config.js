import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	optimizeDeps: {
		exclude: ['@duckdb/duckdb-wasm']
	},
	server: {
		hmr: false,
		proxy: {
			'/api': 'http://localhost:7860',
			'/data': 'http://localhost:7860',
			'/sse': 'http://localhost:7860'
		}
	}
});
