uv run pelican content -o output -s pelicanconf.py
uv run ghp-import output -b gh-pages
git push origin gh-pages