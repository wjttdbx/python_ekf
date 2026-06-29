from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
GENERATED = REPORT_DIR / "generated"
SOURCE = REPORT_DIR / "course_report.md"
DOCX = REPORT_DIR / "基于统一SDC矩阵的仅测角EKF-SDRE航天器相对导航与控制研究.docx"
PDF = REPORT_DIR / "基于统一SDC矩阵的仅测角EKF-SDRE航天器相对导航与控制研究.pdf"

FIGURES = [
    "fig1_lvlh_frame",
    "fig2_flow_diagram",
    "fig3_prediction_error",
    "fig6_noise_sensitivity",
    "fig7_eccentricity_sweep",
    "fig8_monte_carlo",
    "fig9_monte_carlo_ekf",
]


def run(cmd: list[str], cwd: Path = REPORT_DIR) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"Required tool not found on PATH: {name}")
    if name == "pdftoppm":
        found_path = Path(found)
        bundled_poppler = (
            found_path.parent.parent
            / "native"
            / "poppler"
            / "Library"
            / "bin"
            / "pdftoppm.exe"
        )
        if bundled_poppler.exists():
            return str(bundled_poppler)
    return found


def convert_figures() -> None:
    pdftoppm = require_tool("pdftoppm")
    GENERATED.mkdir(parents=True, exist_ok=True)
    for stem in FIGURES:
        src = ROOT / "docs" / "figures" / f"{stem}.pdf"
        if not src.exists():
            raise FileNotFoundError(src)
        out_prefix = GENERATED / stem
        out_png = GENERATED / f"{stem}.png"
        if out_png.exists() and out_png.stat().st_mtime >= src.stat().st_mtime:
            continue
        run([pdftoppm, "-png", "-singlefile", "-r", "180", str(src), str(out_prefix)])


def build_docx() -> None:
    pandoc = require_tool("pandoc")
    run(
        [
            pandoc,
            str(SOURCE.name),
            "--from",
            "markdown+tex_math_dollars+raw_tex",
            "--to",
            "docx",
            "--standalone",
            "--toc",
            "--toc-depth=3",
            "--number-sections",
            "--resource-path",
            str(REPORT_DIR),
            "--output",
            str(DOCX),
        ]
    )


def build_pdf() -> None:
    pandoc = require_tool("pandoc")
    require_tool("xelatex")
    run(
        [
            pandoc,
            str(SOURCE.name),
            "--from",
            "markdown+tex_math_dollars+raw_tex",
            "--standalone",
            "--toc",
            "--toc-depth=3",
            "--number-sections",
            "--pdf-engine=xelatex",
            "-V",
            "documentclass=ctexart",
            "-V",
            "papersize=a4",
            "-V",
            "geometry:margin=2.5cm",
            "-V",
            "linestretch=1.25",
            "-V",
            "colorlinks=true",
            "-V",
            "linkcolor=black",
            "-V",
            "urlcolor=blue",
            "--resource-path",
            str(REPORT_DIR),
            "--output",
            str(PDF),
        ]
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    convert_figures()
    build_docx()
    build_pdf()
    print(f"DOCX: {DOCX}")
    print(f"PDF:  {PDF}")


if __name__ == "__main__":
    main()
