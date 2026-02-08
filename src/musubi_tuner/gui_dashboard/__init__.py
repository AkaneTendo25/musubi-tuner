from musubi_tuner.gui_dashboard.metrics_writer import MetricsWriter
from musubi_tuner.gui_dashboard.server import start_server


def create_metrics_writer(run_dir: str, flush_every: int = 10) -> MetricsWriter:
    return MetricsWriter(run_dir, flush_every=flush_every)


def start_gui_server(run_dir: str, host: str = "0.0.0.0", port: int = 7860):
    return start_server(run_dir, host=host, port=port)
