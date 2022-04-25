PAPER=main

default: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex *.tex figures/*.png figures/*.PNG figures/*.pdf figures/*.pptx
	pdflatex $(PAPER).tex </dev/null
	bibtex $(PAPER)
	pdflatex $(PAPER).tex </dev/null >/dev/null
	pdflatex $(PAPER).tex </dev/null >/dev/null
	pdflatex $(PAPER).tex </dev/null

tidy:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).brf $(PAPER).btx $(PAPER).log $(PAPER).out

clean: tidy
	rm -f $(PAPER).pdf
