"""Entry point; see oplab/remote.py. Run from ml/ with: uv run scripts/remote.py --help"""

try:
    from oplab.remote import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/remote.py")

if __name__ == "__main__":
    main()
