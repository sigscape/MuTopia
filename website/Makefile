# Makefile for Sphinx documentation
# Minimal helpers; `make html` builds the site.

SPHINXOPTS    = --keep-going
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
TUTORIALDIR   = ../tutorials

.PHONY: help clean html sync-notebooks

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

sync-notebooks:
	@echo "Syncing notebooks from $(TUTORIALDIR) to $(SOURCEDIR)/tutorials..."
	@cp $(TUTORIALDIR)/*.ipynb $(SOURCEDIR)/tutorials/

html: sync-notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
