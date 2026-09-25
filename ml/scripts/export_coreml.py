"""Entry point; see oplab/export.py. Run from ml/ with: uv run scripts/export_coreml.py --help"""

try:
    from oplab.export import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/export_coreml.py")

if __name__ == "__main__":
    main()
