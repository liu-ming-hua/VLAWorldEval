"""Check generated website assets and internal links without network access."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for name in ("href", "src", "poster"):
            if attrs.get(name):
                self.targets.append(attrs[name])


def validate_site(directory):
    directory = Path(directory).resolve()
    if not (directory / "index.html").is_file() or not (directory / "searchindex.js").is_file():
        raise ValueError("Missing website homepage or search index")
    documents = {}
    for page in directory.rglob("*.html"):
        if page.relative_to(directory).parts[0] in {"_static", "_images", "assets"}:
            continue  # Theme support templates are not rendered HTML pages.
        parser = Links()
        parser.feed(page.read_text())
        documents[page] = parser
    errors, checked = [], 0
    for page, parser in documents.items():
        for target in parser.targets:
            url = urlsplit(target)
            if url.scheme or url.netloc or target.startswith("/"):
                continue
            resolved = (page.parent / unquote(url.path)).resolve() if url.path else page
            if resolved.is_dir():
                resolved /= "index.html"
            if not resolved.is_relative_to(directory) or not resolved.is_file():
                errors.append(f"{page.relative_to(directory)} -> {target}")
            elif url.fragment and resolved in documents and unquote(url.fragment) not in documents[resolved].ids:
                errors.append(f"Missing anchor: {page.relative_to(directory)} -> {target}")
            checked += 1
    for language in ("en", "zh"):
        for page in (directory / language).glob("*.html"):
            other = "zh" if language == "en" else "en"
            if f'../{other}/{page.name}' not in page.read_text():
                errors.append(f"Missing language counterpart: {page.relative_to(directory)}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"html_pages": len(documents), "internal_links": checked, "network_required": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(validate_site(parser.parse_args().directory)))
