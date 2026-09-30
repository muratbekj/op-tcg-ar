"""Entry point; see oplab/embeddings.py. Run from ml/ with: uv run scripts/generate_embeddings.py --help"""

try:
    from oplab.embeddings import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/generate_embeddings.py")

if __name__ == "__main__":
    main()
