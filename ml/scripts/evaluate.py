"""Entry point; see oplab/evaluate.py. Run from ml/ with: uv run scripts/evaluate.py --help"""

try:
    from oplab.evaluate import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/evaluate.py")

if __name__ == "__main__":
    main()
