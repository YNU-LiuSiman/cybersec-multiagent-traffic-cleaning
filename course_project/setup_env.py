"""One-click environment & data check script for classmates reproducing course_project."""
from pathlib import Path
import importlib
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REQUIRED_PKGS = ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "pyarrow", "requests", "openai"]
REQUIRED_DATA = ["unsw_train.parquet", "unsw_test.parquet"]
OPTIONAL_DATA = ["cic2017_Friday-DDos.csv", "cic2018_head.csv"]


def main() -> int:
    print(f"[1/4] Python: {sys.version.split()[0]} ({sys.executable})")

    missing_pkgs = []
    for pkg in REQUIRED_PKGS:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_pkgs.append("scikit-learn" if pkg == "sklearn" else pkg)

    if missing_pkgs:
        print(f"[2/4] Missing packages: {', '.join(missing_pkgs)}")
        print(f"      Run: pip install {' '.join(missing_pkgs)}")
    else:
        print("[2/4] All required Python packages are installed.")

    missing_data = [f for f in REQUIRED_DATA if not (DATA_DIR / f).exists()]
    if missing_data:
        print(f"[3/4] Missing required data in {DATA_DIR}: {missing_data}")
    else:
        print("[3/4] Core UNSW-NB15 dataset files found.")
    for f in OPTIONAL_DATA:
        status = "found" if (DATA_DIR / f).exists() else "missing (optional)"
        print(f"      - {f}: {status}")

    has_key = bool(os.environ.get("LLM_API_KEY") or (ROOT / ".env").exists() or (Path(__file__).parent / ".env").exists())
    if has_key:
        print("[4/4] LLM API configuration detected.")
    else:
        print("[4/4] LLM_API_KEY not set (only needed to re-run live LLM agent negotiation).")
        print("      PowerShell example:")
        print('        $env:LLM_API_KEY="sk-your-key"')
        print('        $env:LLM_BASE_URL="https://api.openai-next.com/v1"')
        print('        $env:LLM_MODEL="deepseek-v4-1-flash-261017"')

    return 1 if (missing_pkgs or missing_data) else 0


if __name__ == "__main__":
    raise SystemExit(main())
