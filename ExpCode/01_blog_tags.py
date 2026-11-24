"""Lab 1: Blog page demonstrating HTML title, anchor, heading, and related tags.
WordPress itself needs a local server. This program builds a real blog HTML page
and prints every tag it uses, which is what the screenshot captures.
"""
from pathlib import Path

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My Lab Blog</title>
</head>
<body>
  <h1>My Lab Blog</h1>
  <p>This post shows the tags asked in the lab.</p>
  <h2>Anchor tag</h2>
  <p>Visit <a href="https://wordpress.org" title="WordPress home">WordPress</a>
     or jump to the <a href="#notes">notes section</a>.</p>
  <h2 id="notes">Notes</h2>
  <ul>
    <li><strong>title</strong> tag sets the browser tab name.</li>
    <li><strong>a</strong> (anchor) tag creates a link.</li>
    <li><strong>h1</strong> and <strong>h2</strong> are headings.</li>
  </ul>
  <img src="banner.png" alt="Blog banner" width="320" height="80">
</body>
</html>
"""

def main():
    out = Path(__file__).resolve().parent.parent / "blog_demo.html"
    out.write_text(HTML, encoding="utf-8")
    print("=== Program 01: Blog tags (title, anchor, headings) ===")
    print(f"Wrote HTML blog page: {out.name}")
    print()
    print("Tags demonstrated:")
    tags = [
        ("<title>", "sets the page title shown in the browser tab"),
        ("<a href>", "anchor tag — external link and in-page link"),
        ("<h1>", "main heading of the blog post"),
        ("<h2>", "section headings"),
        ("<p>", "paragraph text"),
        ("<ul>/<li>", "bullet list"),
        ("<strong>", "bold emphasis"),
        ("<img alt>", "image with required alt text"),
        ("id + href=\"#notes\"", "anchor target inside the same page"),
    ]
    for tag, why in tags:
        print(f"  {tag:22} {why}")
    print()
    print("Sample anchor lines from the page:")
    for line in HTML.splitlines():
        if "<a " in line or "<title>" in line or "<h1>" in line:
            print(" ", line.strip())
    print()
    print("Open blog_demo.html in a browser to see the page.")
    print("Program completed successfully.")

if __name__ == "__main__":
    main()
