<script>
	import { onMount } from 'svelte';
	import { processStatuses } from '$lib/stores/processes.js';

	let management = $state(null);
	let loading = $state(true);
	let error = $state('');
	let actionMessage = $state('');
	let actionTone = $state('muted');
	let openingSetup = $state(false);
	let openingRepo = $state(false);
	let selectedBranch = $state('ltx-2');

	let processesBusy = $derived.by(() => {
		const statuses = $processStatuses || {};
		return Object.values(statuses).some((status) => status && (status.state === 'running' || status.state === 'stopping'));
	});

	function toneColor(tone) {
		if (tone === 'success') return 'var(--success)';
		if (tone === 'danger') return 'var(--danger)';
		if (tone === 'warning') return 'var(--warning)';
		return 'var(--text-secondary)';
	}

	function toneBg(tone) {
		if (tone === 'success') return 'var(--success-muted, rgba(34,197,94,0.10))';
		if (tone === 'danger') return 'var(--danger-muted)';
		if (tone === 'warning') return 'color-mix(in srgb, var(--warning) 12%, transparent)';
		return 'var(--bg-elevated)';
	}

	function statusTone(status) {
		if (status === 'ok') return 'success';
		if (status === 'error') return 'danger';
		if (status === 'warning') return 'warning';
		return 'muted';
	}

	function doctorTone(doctor) {
		if (!doctor) return 'muted';
		if (doctor.error_count > 0) return 'danger';
		if (doctor.warning_count > 0) return 'warning';
		return 'success';
	}

	function relatedPageLabel(page) {
		if (page === '/') return 'Open Projects';
		if (page === '/dataset') return 'Open Dataset';
		if (page === '/caching') return 'Open Caching';
		if (page === '/training') return 'Open Training';
		if (page === '/inference') return 'Open Inference';
		return 'Open related page';
	}

	function showRelatedPage(page) {
		return Boolean(page) && page !== '/settings';
	}

	function statusDotColor(status) {
		return toneColor(statusTone(status));
	}

	function repoTone(repo) {
		if (!repo) return 'muted';
		if (repo.dirty || repo.diverged) return 'warning';
		if (repo.update_available) return 'accent';
		if (repo.exists && repo.is_git_repo) return 'success';
		return 'muted';
	}

	function accentTone(tone) {
		if (tone === 'accent') return 'var(--accent)';
		return toneColor(tone);
	}

	function accentBg(tone) {
		if (tone === 'accent') return 'var(--accent-muted)';
		return toneBg(tone);
	}

	async function readJsonOrError(res, fallbackMessage) {
		const contentType = res.headers.get('content-type') || '';
		if (contentType.includes('application/json')) {
			return await res.json();
		}

		const body = await res.text().catch(() => '');
		if (body.toLowerCase().includes('<!doctype') || body.toLowerCase().includes('<html')) {
			throw new Error(`${fallbackMessage} The dashboard backend likely needs to be restarted so the new management API becomes available.`);
		}
		throw new Error(fallbackMessage);
	}

	async function refresh() {
		loading = true;
		error = '';
		actionMessage = '';
		try {
			const res = await fetch('/api/system/management-status', { cache: 'no-store' });
			if (!res.ok) {
				const payload = await readJsonOrError(res, 'Failed to load management status.');
				throw new Error(payload?.detail || 'Failed to load management status.');
			}
			management = await readJsonOrError(res, 'Failed to load management status.');
			if (management?.branch === 'ltx-2' || management?.branch === 'ltx-2-dev') {
				selectedBranch = management.branch;
			}
		} catch (e) {
			error = e.message || 'Failed to load management status';
		}
		loading = false;
	}

	async function postAction(url, which) {
		const setter = which === 'setup' ? (v) => openingSetup = v : (v) => openingRepo = v;
		setter(true);
		actionMessage = '';
		try {
			const res = await fetch(url, { method: 'POST' });
			const payload = await readJsonOrError(res, 'Management action failed.').catch((err) => {
				if (!res.ok) throw err;
				return {};
			});
			if (!res.ok) {
				throw new Error(payload.detail || 'Action failed');
			}
			actionTone = 'success';
			actionMessage = which === 'setup'
				? 'Setup / Update opened in a new window.'
				: 'Repository folder opened.';
		} catch (e) {
			actionTone = 'danger';
			actionMessage = e.message || 'Action failed';
		}
		setter(false);
		await refresh();
	}

	onMount(refresh);
</script>

<div class="space-y-5">
	<div>
		<h2 class="text-base font-semibold" style="color: var(--text-primary);">Setup & Updates</h2>
		<p class="text-[12px]" style="color: var(--text-muted);">
			Manage this install from the dashboard: check repo/update status, confirm shortcuts still exist, and reopen the same Setup / Update tool used for first-time installation.
		</p>
	</div>

	{#if actionMessage}
		<div class="px-3 py-2 text-[12px]" style="color: {accentTone(actionTone)}; background: {accentBg(actionTone)}; border-radius: var(--radius-sm); border: 1px solid color-mix(in srgb, {accentTone(actionTone)} 28%, var(--border));">
			{actionMessage}
		</div>
	{/if}

	{#if processesBusy}
		<div class="px-3 py-2 text-[12px]" style="color: var(--warning); background: color-mix(in srgb, var(--warning) 12%, transparent); border-radius: var(--radius-sm); border: 1px solid color-mix(in srgb, var(--warning) 28%, var(--border));">
			Setup / Update is disabled while caching, training, or inference processes are running.
		</div>
	{/if}

	{#if loading}
		<div class="p-5 text-[12px]" style="color: var(--text-muted); background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
			Loading management status...
		</div>
	{:else if error}
		<div class="p-5 text-[12px]" style="color: var(--danger); background: var(--danger-muted); border: 1px solid color-mix(in srgb, var(--danger) 30%, var(--border)); border-radius: var(--radius-md);">
			{error}
		</div>
	{:else if management}
		<div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0.45;"></div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Repository</div>
				<div class="space-y-2">
					<div class="px-3 py-2 text-[12px]" style="color: {accentTone(repoTone(management.repo))}; background: {accentBg(repoTone(management.repo))}; border-radius: var(--radius-sm);">
						{management.repo.summary}
					</div>
					<div class="grid grid-cols-2 gap-3 text-[12px]">
						<div>
							<div style="color: var(--text-muted);">Repo Root</div>
							<div class="font-mono break-all" style="color: var(--text-primary);">{management.repo_root}</div>
						</div>
						<div>
							<div style="color: var(--text-muted);">Configured Branch</div>
							<div class="font-mono" style="color: var(--text-primary);">{management.branch}</div>
						</div>
						<div>
							<div style="color: var(--text-muted);">Checked Out Branch</div>
							<div class="font-mono" style="color: var(--text-primary);">{management.repo.branch || '--'}</div>
						</div>
						<div>
							<div style="color: var(--text-muted);">Current Commit</div>
							<div class="font-mono" style="color: var(--text-primary);">{management.repo.head_short || '--'}</div>
						</div>
						<div>
							<div style="color: var(--text-muted);">Remote Commit</div>
							<div class="font-mono" style="color: var(--text-primary);">{management.repo.remote_head_short || '--'}</div>
						</div>
						<div>
							<div style="color: var(--text-muted);">Dirty Worktree</div>
							<div style="color: {management.repo.dirty ? 'var(--warning)' : 'var(--success)'};">{management.repo.dirty ? 'Yes' : 'No'}</div>
						</div>
						<div>
							<div style="color: var(--text-muted);">Remote Ahead</div>
							<div style="color: var(--text-primary);">{management.repo.remote_ahead_count}</div>
						</div>
						<div class="col-span-2">
							<div style="color: var(--text-muted);">Remote URL</div>
							<div class="font-mono break-all" style="color: var(--text-primary);">{management.remote_url || '--'}</div>
						</div>
					</div>
				</div>
			</div>

			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); position: relative; overflow: hidden;">
				<div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0.45;"></div>
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Install Health</div>
				<div class="grid grid-cols-2 gap-3 text-[12px]">
					<div>
						<div style="color: var(--text-muted);">Last Successful Setup</div>
						<div style="color: var(--text-primary);">{management.install.last_success_label}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Install State</div>
						<div style="color: {management.install_state_present ? 'var(--success)' : 'var(--warning)'};">{management.install_state_present ? 'Recorded' : 'Missing'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Virtual Env</div>
						<div style="color: {management.install.venv_exists ? 'var(--success)' : 'var(--danger)'};">{management.install.venv_exists ? 'Ready' : 'Missing'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Frontend Dist</div>
						<div style="color: {management.install.frontend_dist_exists ? 'var(--success)' : 'var(--danger)'};">{management.install.frontend_dist_exists ? 'Built' : 'Missing'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Dependencies</div>
						<div style="color: {management.install.deps_recommended ? 'var(--warning)' : 'var(--success)'};">{management.install.deps_recommended ? 'Review Recommended' : 'Looks Current'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Frontend Refresh</div>
						<div style="color: {management.install.frontend_recommended ? 'var(--warning)' : 'var(--success)'};">{management.install.frontend_recommended ? 'Rebuild Recommended' : 'Looks Current'}</div>
					</div>
					<div class="col-span-2">
						<div style="color: var(--text-muted);">Venv Python</div>
						<div class="font-mono break-all" style="color: var(--text-primary);">{management.install.venv_python_path}</div>
					</div>
				</div>
			</div>

			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Launchers & Shortcuts</div>
				<div class="grid grid-cols-2 gap-3 text-[12px]">
					<div>
						<div style="color: var(--text-muted);">Dashboard Launcher</div>
						<div style="color: {management.install.dashboard_launcher_exists ? 'var(--success)' : 'var(--warning)'};">{management.install.dashboard_launcher_exists ? 'Present' : 'Missing'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Setup Launcher</div>
						<div style="color: {management.install.setup_launcher_exists ? 'var(--success)' : 'var(--warning)'};">{management.install.setup_launcher_exists ? 'Present' : 'Missing'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Dashboard Shortcut</div>
						<div style="color: {management.install.dashboard_shortcut_exists ? 'var(--success)' : 'var(--warning)'};">{management.install.dashboard_shortcut_exists ? 'Present' : 'Missing'}</div>
					</div>
					<div>
						<div style="color: var(--text-muted);">Setup Shortcut</div>
						<div style="color: {management.install.setup_shortcut_exists ? 'var(--success)' : 'var(--warning)'};">{management.install.setup_shortcut_exists ? 'Present' : 'Missing'}</div>
					</div>
					<div class="col-span-2">
						<div style="color: var(--text-muted);">Setup Launch Mode</div>
						<div style="color: var(--text-primary);">{management.actions.setup_launch_mode || '--'}</div>
					</div>
				</div>
				<div class="space-y-1 text-[11px]" style="color: var(--text-muted);">
					<div class="font-mono break-all">{management.install.dashboard_shortcut_path}</div>
					<div class="font-mono break-all">{management.install.setup_shortcut_path}</div>
				</div>
			</div>

			<div class="p-5 space-y-4" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Actions</div>
				<div class="space-y-2">
					<div class="text-[11px]" style="color: var(--text-muted);">Target Branch</div>
					<div class="flex flex-wrap gap-2">
						<button
							type="button"
							onclick={() => selectedBranch = 'ltx-2'}
							class="px-3 py-2 text-[12px] font-medium"
							style="background: {selectedBranch === 'ltx-2' ? 'var(--sidebar-active)' : 'var(--bg-elevated)'}; color: {selectedBranch === 'ltx-2' ? 'var(--accent)' : 'var(--text-secondary)'}; border: 1px solid {selectedBranch === 'ltx-2' ? 'color-mix(in srgb, var(--accent) 36%, var(--border))' : 'var(--border)'}; border-radius: var(--radius-sm);"
						>
							ltx-2
						</button>
						<button
							type="button"
							onclick={() => selectedBranch = 'ltx-2-dev'}
							class="px-3 py-2 text-[12px] font-medium"
							style="background: {selectedBranch === 'ltx-2-dev' ? 'var(--sidebar-active)' : 'var(--bg-elevated)'}; color: {selectedBranch === 'ltx-2-dev' ? 'var(--accent)' : 'var(--text-secondary)'}; border: 1px solid {selectedBranch === 'ltx-2-dev' ? 'color-mix(in srgb, var(--accent) 36%, var(--border))' : 'var(--border)'}; border-radius: var(--radius-sm);"
						>
							ltx-2-dev
						</button>
					</div>
					<div class="grid grid-cols-1 gap-2 text-[11px]" style="color: var(--text-muted);">
						<div class="px-3 py-2" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);">
							<span class="font-mono" style="color: var(--text-primary);">ltx-2</span> keeps you on the steadier branch for normal usage.
						</div>
						<div class="px-3 py-2" style="background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);">
							<span class="font-mono" style="color: var(--text-primary);">ltx-2-dev</span> tracks newer features and changes sooner.
						</div>
					</div>
				</div>
				<div class="flex flex-wrap gap-2">
					<button
						type="button"
						onclick={() => postAction(`/api/system/management/open-setup?branch=${encodeURIComponent(selectedBranch)}`, 'setup')}
						disabled={openingSetup || processesBusy || !management.actions.can_launch_setup}
						class="px-3 py-2 text-[12px] font-semibold disabled:opacity-40"
						style="background: color-mix(in srgb, var(--accent) 74%, var(--bg-elevated)); color: var(--text-primary); border: 1px solid color-mix(in srgb, var(--accent) 36%, var(--border)); border-radius: var(--radius-sm);"
					>
						{openingSetup ? 'Opening...' : `Open Setup / Update (${selectedBranch})`}
					</button>
					<button
						type="button"
						onclick={() => postAction('/api/system/management/open-repo', 'repo')}
						disabled={openingRepo || !management.actions.can_open_repo}
						class="px-3 py-2 text-[12px] font-medium disabled:opacity-40"
						style="background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
					>
						{openingRepo ? 'Opening...' : 'Open Repo Folder'}
					</button>
					<button
						type="button"
						onclick={refresh}
						disabled={loading}
						class="px-3 py-2 text-[12px] font-medium disabled:opacity-40"
						style="background: var(--bg-elevated); color: var(--text-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm);"
					>
						Refresh Status
					</button>
				</div>
				<div class="text-[11px]" style="color: var(--text-muted);">
					Use Setup / Update for repository sync, branch switching, dependency repair, frontend rebuilds, and shortcut recreation. The dashboard intentionally does not mutate the environment directly.
				</div>
			</div>
		</div>

		{#if management.doctor}
			<div class="p-4 space-y-3" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
				<div class="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
					<div class="space-y-2">
						<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Ready Check</div>
						<div class="text-[13px] font-semibold" style="color: var(--text-primary);">
							{management.doctor.loaded_project ? (management.doctor.project_name || 'Current Project') : 'Environment Readiness'}
						</div>
						<div class="text-[11px]" style="color: var(--text-secondary);">
							{management.doctor.summary}
						</div>
					</div>
					<div class="text-[11px]" style="color: {accentTone(doctorTone(management.doctor))};">
						{management.doctor.error_count} blocking issue(s) • {management.doctor.warning_count} warning(s)
					</div>
				</div>

				<div class="space-y-3">
					<div class="space-y-1.5">
						<div class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Environment</div>
						<div style="background: color-mix(in srgb, var(--bg-elevated) 88%, transparent); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden;">
							{#each management.doctor.environment_checks as item, index}
								<div class="px-3 py-2" style="border-bottom: {index < management.doctor.environment_checks.length - 1 ? '1px solid var(--border-subtle)' : 'none'};">
									<div class="flex items-start gap-2.5">
										<span class="w-2 h-2 rounded-full flex-shrink-0 mt-1.5" style="background: {statusDotColor(item.status)};"></span>
										<div class="min-w-0 flex-1 space-y-0.5">
											<div class="flex flex-wrap items-center gap-x-2 gap-y-1">
												<div class="text-[11px] font-medium" style="color: var(--text-primary);">{item.label}</div>
												{#if showRelatedPage(item.page)}
													<a href={item.page} class="text-[10px]" style="color: var(--accent);">{relatedPageLabel(item.page)}</a>
												{/if}
											</div>
											<div class="text-[10px]" style="color: var(--text-secondary);">{item.detail}</div>
										</div>
									</div>
								</div>
							{/each}
						</div>
					</div>

					{#if management.doctor.loaded_project}
						<div class="space-y-1.5">
							<div class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Project Assets</div>
							<div style="background: color-mix(in srgb, var(--bg-elevated) 88%, transparent); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden;">
								{#each management.doctor.asset_checks as item, index}
									<div class="px-3 py-2" style="border-bottom: {index < management.doctor.asset_checks.length - 1 ? '1px solid var(--border-subtle)' : 'none'};">
										<div class="flex items-start gap-2.5">
											<span class="w-2 h-2 rounded-full flex-shrink-0 mt-1.5" style="background: {statusDotColor(item.status)};"></span>
											<div class="min-w-0 flex-1 space-y-0.5">
												<div class="flex flex-wrap items-center gap-x-2 gap-y-1">
													<div class="text-[11px] font-medium" style="color: var(--text-primary);">{item.label}</div>
													{#if showRelatedPage(item.page)}
														<a href={item.page} class="text-[10px]" style="color: var(--accent);">{relatedPageLabel(item.page)}</a>
													{/if}
												</div>
												<div class="text-[10px]" style="color: var(--text-secondary);">{item.detail}</div>
											</div>
										</div>
									</div>
								{/each}
							</div>
						</div>

						<div class="space-y-1.5">
							<div class="text-[10px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Launch Validation</div>
							<div style="background: color-mix(in srgb, var(--bg-elevated) 88%, transparent); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden;">
								{#each management.doctor.processes as item, index}
									<div class="px-3 py-2" style="border-bottom: {index < management.doctor.processes.length - 1 ? '1px solid var(--border-subtle)' : 'none'};">
										<div class="flex items-start gap-2.5">
											<span class="w-2 h-2 rounded-full flex-shrink-0 mt-1.5" style="background: {statusDotColor(item.status)};"></span>
											<div class="min-w-0 flex-1 space-y-0.5">
												<div class="flex flex-wrap items-center gap-x-2 gap-y-1">
													<div class="text-[11px] font-medium" style="color: var(--text-primary);">{item.label}</div>
													{#if showRelatedPage(item.page)}
														<a href={item.page} class="text-[10px]" style="color: var(--accent);">{relatedPageLabel(item.page)}</a>
													{/if}
												</div>
												<div class="text-[10px]" style="color: var(--text-secondary);">{item.summary}</div>
											</div>
										</div>
									</div>
								{/each}
							</div>
						</div>
					{:else}
						<div class="px-3 py-2 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between" style="background: color-mix(in srgb, var(--bg-elevated) 88%, transparent); border: 1px solid var(--border); border-radius: var(--radius-sm);">
							<div class="space-y-1">
								<div class="text-[11px] font-medium" style="color: var(--text-primary);">Project checks are not active yet</div>
								<div class="text-[10px]" style="color: var(--text-secondary);">
									Load an existing project or create a new one to unlock asset checks and launch validation for caching, training, and inference.
								</div>
							</div>
							<a href="/" class="px-3 py-1.5 text-[10px] font-medium" style="background: color-mix(in srgb, var(--accent) 10%, var(--bg-surface)); color: var(--text-primary); border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--border)); border-radius: var(--radius-sm);">
								Open Projects
							</a>
						</div>
					{/if}
				</div>
			</div>
		{/if}

		{#if management.recommendations?.length}
			<div class="p-5 space-y-2" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--text-muted); font-family: var(--font-label);">Recommended Next Steps</div>
				{#each management.recommendations as item}
					<div class="px-3 py-2 text-[12px]" style="background: var(--bg-elevated); border-radius: var(--radius-sm); color: var(--text-secondary);">
						{item}
					</div>
				{/each}
			</div>
		{/if}

		{#if management.warnings?.length}
			<div class="p-5 space-y-2" style="background: var(--bg-surface); border: 1px solid color-mix(in srgb, var(--warning) 24%, var(--border)); border-radius: var(--radius-md);">
				<div class="text-[11px] font-semibold uppercase tracking-wider" style="color: var(--warning); font-family: var(--font-label);">Warnings</div>
				{#each management.warnings as item}
					<div class="px-3 py-2 text-[12px]" style="background: color-mix(in srgb, var(--warning) 10%, transparent); border-radius: var(--radius-sm); color: var(--text-secondary);">
						{item}
					</div>
				{/each}
			</div>
		{/if}
	{/if}
</div>
