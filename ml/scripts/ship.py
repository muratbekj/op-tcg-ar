"""Entry point; see oplab/shipped.py. Run from ml/ with: uv run scripts/ship.py --help"""

try:
    from oplab.shipped import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/ship.py")

if __name__ == "__main__":
    main()
