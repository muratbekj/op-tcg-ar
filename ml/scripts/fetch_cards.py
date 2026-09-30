"""Entry point; see oplab/fetch.py. Run from ml/ with: uv run scripts/fetch_cards.py --help"""

try:
    from oplab.fetch import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/fetch_cards.py")

if __name__ == "__main__":
    main()
