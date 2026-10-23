"""Offline validation of current documentation's source-relative links."""
from pathlib import Path
import re


ASSET_DIRECTORIES = {"checkpoints", "datasets", "sample_data", "environment", "results", "third_party", "wheelhouse", ".cache", ".archive"}


def validate_documentation(root):
    root = Path(root).resolve()
    files = [root / name for name in ("README.md", "README_CN.md", "CONTRIBUTING.md", "LICENSE_SCOPE.md")]
    # Model onboarding documents may live under docs/models/. Archived plans
    # intentionally retain historical links and are not current instructions.
    files += [p for p in (root / "docs").rglob("*.md")
              if "archive" not in p.relative_to(root / "docs").parts]
    files += [root / name / "README.md" for name in ("adapters", "cli", "evaluation", "scripts", "configs")]
    if (root / "tests/README.md").is_file():
        files.append(root / "tests/README.md")
    errors, checked = [], 0
    for file in files:
        if not file.is_file():
            errors.append(f"Missing current documentation: {file.relative_to(root)}")
            continue
        text = re.sub(r"```[\s\S]*?```", "", file.read_text())
        hrefs = re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", text)
        hrefs += re.findall(r'''<(?:img|a|video|source)\b[^>]*?\b(?:src|href)=["']([^"']+)["']''', text,
                            flags=re.IGNORECASE)
        for href in hrefs:
            if href.startswith(("#", "/")) or re.match(r"[a-zA-Z]+:", href):
                continue
            href = href.split("#", 1)[0].strip("<>")
            target = (file.parent / href).resolve()
            if not target.is_relative_to(root):
                errors.append(f"Link escapes source: {file.relative_to(root)} -> {href}")
                continue
            if target.relative_to(root).parts and target.relative_to(root).parts[0] in ASSET_DIRECTORIES:
                continue  # Separately obtained assets are not a source-install prerequisite.
            checked += 1
            if not target.exists():
                errors.append(f"Broken source link: {file.relative_to(root)} -> {href}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"documents": len(files), "source_links": checked, "external_links_checked": False}
