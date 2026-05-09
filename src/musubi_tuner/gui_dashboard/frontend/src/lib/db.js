import * as duckdb from '@duckdb/duckdb-wasm';

let db = null;
let conn = null;

export async function getConnection() {
	if (conn) return conn;

	const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
	const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

	const worker_url = URL.createObjectURL(
		new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' })
	);

	const worker = new Worker(worker_url);
	const logger = new duckdb.ConsoleLogger();
	db = new duckdb.AsyncDuckDB(logger, worker);
	await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
	conn = await db.connect();

	URL.revokeObjectURL(worker_url);
	return conn;
}

export async function loadParquet(url) {
	const c = await getConnection();
	const response = await fetch(url, { cache: 'no-store' });
	if (!response.ok || response.status === 204) return null;

	const buffer = await response.arrayBuffer();
	if (buffer.byteLength === 0) return null;

	try {
		await db.dropFile('metrics.parquet');
	} catch {
		// ignore if not registered yet
	}
	await db.registerFileBuffer('metrics.parquet', new Uint8Array(buffer));
	await c.query(`CREATE OR REPLACE TABLE metrics AS SELECT * FROM read_parquet('metrics.parquet')`);
	return c;
}

export async function query(sql) {
	const c = await getConnection();
	const result = await c.query(sql);
	return result.toArray().map((row) => row.toJSON());
}
