#!/bin/bash
# Build the paper.  Tectonic downloads whatever TeX packages it needs on first use.
#   bash build.sh          build main.pdf
#   bash build.sh clean    remove build products
set -e
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-tectonic}
if [ "$1" = clean ]; then
    rm -f main.pdf main.aux main.bbl main.blg main.log main.out
    exit 0
fi
"$TECTONIC" -X compile main.tex --keep-intermediates --synctex
if command -v pdfinfo >/dev/null; then
    echo "pages: $(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
elif command -v pdftotext >/dev/null; then
    echo "pages: $(pdftotext -layout main.pdf - | tr -cd '\f' | wc -c | tr -d ' ')"
fi
