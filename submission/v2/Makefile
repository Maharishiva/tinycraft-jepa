TECTONIC ?= tectonic
SOURCES := main.tex $(wildcard sections/*.tex) $(wildcard tables/*.tex) references.bib iclr2027_conference.sty iclr2027_conference.bst $(wildcard ../shared-v2/figures/*.pdf)

.PHONY: all clean

all: main.pdf

main.pdf: $(SOURCES)
	$(TECTONIC) -X compile main.tex --keep-intermediates --synctex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.synctex.gz
