from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Predictive_Text_Markov_Chain_Project.zip"
PROJECT_FOLDER = "Predictive_Text_Markov_Project"
INCLUDED_DIRS = ("backend", "frontend", "data", "tests", "documentation", "presentation", "screenshots")
INCLUDED_FILES = ("README.md", "requirements.txt", ".gitignore")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "node_modules", ".git"}


def project_files():
    files = [ROOT / name for name in INCLUDED_FILES]
    for dirname in INCLUDED_DIRS:
        folder = ROOT / dirname
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo", ".DS_Store"}:
                continue
            files.append(path)
    return sorted(path for path in files if path.is_file())


def main():
    required = [ROOT / "screenshots" / "homepage.png", ROOT / "screenshots" / "dashboard.png",
                ROOT / "screenshots" / "mobile.png", ROOT / "documentation" / "Final_Report.pdf",
                ROOT / "presentation" / "Predictive_Text_Markov_Chain_15_Slides.pptx"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Create the required deliverables first: " + ", ".join(missing))

    files = project_files()
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=8) as archive:
        for path in files:
            archive.write(path, f"{PROJECT_FOLDER}/{path.relative_to(ROOT).as_posix()}")

    with ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None, "ZIP integrity check failed"
        assert all(name.startswith(f"{PROJECT_FOLDER}/") for name in archive.namelist())
    print(f"Created {ARCHIVE.name} with {len(files)} project files in {PROJECT_FOLDER}/")


if __name__ == "__main__":
    main()
