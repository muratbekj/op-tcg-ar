"""Entry point; see oplab/train.py. Run from ml/ with: uv run scripts/train_embedding.py --help"""

try:
    from oplab.train import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/train_embedding.py")

if __name__ == "__main__":
    main()
