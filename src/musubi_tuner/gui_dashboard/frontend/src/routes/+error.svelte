<script>
	import { page } from '$app/state';

	const DEV_DASHBOARD_URL = 'http://127.0.0.1:5173';
	const BACKEND_URL = 'http://127.0.0.1:7860';
	const DOCS_URL = 'https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2-dev/docs/ltx_2.md';

	let status = $derived(page.status || 500);
	let message = $derived(page.error?.message || 'The dashboard page could not be loaded.');
</script>

<svelte:head>
	<title>Dashboard could not load</title>
</svelte:head>

<div class="min-h-full flex items-center justify-center px-4 py-10">
	<div class="w-full max-w-xl p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
		<div class="flex items-start justify-between gap-4">
			<div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--danger); font-family: var(--font-label);">Error {status}</div>
				<h1 class="mt-1 text-lg font-semibold" style="color: var(--text-primary);">Dashboard could not load</h1>
			</div>
			<a
				href="/"
				class="px-3 py-1.5 text-[11px] font-medium"
				style="background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
			>
				Projects
			</a>
		</div>

		<div class="space-y-2 text-[12px]" style="color: var(--text-secondary);">
			<p>{message}</p>
			<p>
				The backend API may not be running, or the page may have been opened from the backend static URL instead of the live Vite dashboard.
			</p>
		</div>

		<div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
			<div class="px-3 py-2" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);">
				<div style="color: var(--text-muted);">Live dashboard</div>
				<a href={DEV_DASHBOARD_URL} class="font-mono break-all" style="color: var(--accent);">{DEV_DASHBOARD_URL}</a>
			</div>
			<div class="px-3 py-2" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);">
				<div style="color: var(--text-muted);">Backend API</div>
				<div class="font-mono break-all" style="color: var(--text-primary);">{BACKEND_URL}</div>
			</div>
		</div>

		<div class="flex flex-wrap gap-2">
			<a
				href={DOCS_URL}
				target="_blank"
				rel="noreferrer"
				class="px-3 py-1.5 text-[11px] font-medium"
				style="background: var(--accent-muted); color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 32%, var(--border)); border-radius: var(--radius-sm);"
			>
				LTX-2 docs
			</a>
			<a
				href="/settings"
				class="px-3 py-1.5 text-[11px] font-medium"
				style="background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
			>
				Setup
			</a>
		</div>
	</div>
</div>
