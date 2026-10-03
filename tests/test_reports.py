import json
import re

from src.config import ROOT
from src.reporting.documents import render


def test_readme_and_limitations_match_tables():
    # Both documents are generated from reports/tables; the committed files must equal a fresh render,
    # so every number in them traces to a table produced by the pipeline.
    readme, lim = render()
    assert (ROOT / "README.md").read_text() == readme
    assert (ROOT / "LIMITATIONS.md").read_text() == lim


def test_readme_links_and_figures_exist():
    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)#]+)\)", text):
        assert (ROOT / target).exists(), target


def test_notebook_references_exist():
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for c in json.loads(nb.read_text())["cells"]:
            src = "".join(c["source"])
            for t in re.findall(r'T / "([^"]+)"', src):
                assert (ROOT / "reports" / "tables" / t).exists(), (nb.name, t)
            for f in re.findall(r'show\("([^"]+)"\)', src):
                assert (ROOT / "reports" / "figures" / f).exists(), (nb.name, f)


def test_markdown_word_limits():
    words = lambda p: len((ROOT / p).read_text().split())  # noqa: E731
    assert 300 <= words("reports/shap_interactions.md") <= 500
    assert 400 <= words("reports/impact_analysis.md") <= 600
