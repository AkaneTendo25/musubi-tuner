"""
GUI dashboard tests have been split into tests/gui/ for maintainability.

Run:  python -m pytest tests/gui/ -v

Modules:
  - tests/gui/test_schema.py         — ProjectConfig schema
  - tests/gui/test_toml_export.py    — TOML value serialization & dataset export
  - tests/gui/test_commands.py       — CLI command builders
  - tests/gui/test_process_manager.py — ManagedProcess & ProcessManager
  - tests/gui/test_api.py            — API routers & end-to-end workflows
"""
