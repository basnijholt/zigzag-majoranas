BASE_DIR = ../tmp
CODE_DIR = $(BASE_DIR)/code
PAPER_DIR = $(BASE_DIR)/paper

plot:
	git ls-files *.svg | sed s/svg// | xargs -I % inkscape $$PWD/%svg --export-pdf=%pdf -C
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 ../paper-figures.ipynb

pdf: bbl
	pdflatex zigzag.tex
	pdflatex zigzag.tex

bbl:
	pdflatex zigzag.tex
	bibtex zigzag.aux

clean:
	-rm -rf $(BASE_DIR)

code:
	-mkdir -p $(BASE_DIR)
	-mkdir -p $(CODE_DIR)
	cp ../README.md $(CODE_DIR)/README.txt
	cp ../LICENSE.txt $(CODE_DIR)/
	cp ../environment.yml $(CODE_DIR)/
	cp ../paper-figures.ipynb $(CODE_DIR)/
	cp ../zigzag.py $(CODE_DIR)/

yaml2bib:
	yaml2bib \
	  --bib_fname "zigzag.bib" \
	  --dois_yaml "zigzag.yaml" \
	  --replacements_yaml "replacements.yaml" \
	  --email "bas@nijho.lt" \
	  --static_bib "not_on_crossref.bib"

paper:
	-mkdir -p $(BASE_DIR)
	-mkdir -p $(PAPER_DIR)
	cp zigzag.bib $(PAPER_DIR)
	cp zigzag.tex $(PAPER_DIR)
	export DATE=`date +%Y-%m-%d`
	sed -i '' -e "s/\\\today/`date +%Y-%m-%d`/g" $(PAPER_DIR)/zigzag.tex
	-mkdir -p $(PAPER_DIR)/figures
	cp -r figures/*pdf $(PAPER_DIR)/figures/

data: code
	$(eval TARGET_DIR := $(BASE_DIR)/data_sub)
	-mkdir -p $(TARGET_DIR)
	-mkdir -p $(TARGET_DIR)/data
	cp -R ../data/* $(TARGET_DIR)/data
	-mv $(CODE_DIR)/* $(TARGET_DIR)
	cd $(TARGET_DIR) && zip -Zstore -r ../zigzag_data.zip *

arxiv: paper bbl
	$(eval TARGET_DIR := $(BASE_DIR)/arxiv)
	-mkdir -p $(TARGET_DIR)
	cp zigzag.bbl $(TARGET_DIR)
	-mv $(PAPER_DIR)/* $(TARGET_DIR)
	rm $(TARGET_DIR)/zigzag.bib
	cd $(TARGET_DIR) && zip -r ../zigzag_arxiv.zip *

journal: paper code
	$(eval TARGET_DIR := $(BASE_DIR)/journal)
	-mkdir -p $(TARGET_DIR)
	-mv $(CODE_DIR) $(TARGET_DIR)/supplementary
	-mv $(PAPER_DIR)/* $(TARGET_DIR)
	cd $(TARGET_DIR) && zip -r ../zigzag_journal.zip *
