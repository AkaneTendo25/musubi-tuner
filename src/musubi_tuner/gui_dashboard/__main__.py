"""Entry point for `python -m musubi_tuner.gui_dashboard`."""

import argparse
import logging

import uvicorn

from musubi_tuner.gui_dashboard.management_server import create_management_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Training Manager")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--project", type=str, default=None, help="Path to project.json to load on startup")
    args = parser.parse_args()

    app = create_management_app(project_path=args.project)
    logger.info(f"Starting LTX-2 Training Manager at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
