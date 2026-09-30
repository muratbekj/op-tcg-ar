"""Entry point; see oplab/dataset.py. Run from ml/ with: uv run scripts/prepare_dataset.py --help"""

try:
    from oplab.dataset import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/prepare_dataset.py")

if __name__ == "__main__":
    main()
