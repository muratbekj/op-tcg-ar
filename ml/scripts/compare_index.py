"""Entry point; see oplab/compare_index.py. Run from ml/ with: uv run scripts/compare_index.py A.f32 B.f32"""

try:
    from oplab.compare_index import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/compare_index.py")

if __name__ == "__main__":
    main()
