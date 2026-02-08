import { writable } from 'svelte/store';

export const connected = writable(false);
export const updateTick = writable(0);

let evtSource = null;
let reconnectTimer = null;

export function connectSSE() {
	if (evtSource) return;

	function open() {
		evtSource = new EventSource('/sse');

		evtSource.onopen = () => {
			connected.set(true);
		};

		evtSource.addEventListener('update', () => {
			updateTick.update((n) => n + 1);
		});

		evtSource.onerror = () => {
			connected.set(false);
			evtSource.close();
			evtSource = null;
			reconnectTimer = setTimeout(open, 3000);
		};
	}

	open();
}

export function disconnectSSE() {
	if (evtSource) {
		evtSource.close();
		evtSource = null;
	}
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	connected.set(false);
}
