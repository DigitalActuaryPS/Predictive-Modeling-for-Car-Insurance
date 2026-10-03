"""In Python, ^ is bitwise XOR, not a power. A previous version of this project used it
in model formulas by mistake. No source file may contain the character at all (powers use
**, np.power or splines)."""
from src.config import ROOT


def test_no_caret_anywhere_in_src():
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if "^" in line:
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_scan_covers_source_tree():
    files = list((ROOT / "src").rglob("*.py"))
    assert len(files) >= 20  # guard against the scan silently looking at an empty tree
