BASE_DIR = ../tmp
CODE_DIR = $(BASE_DIR)/code
PAPER_DIR = $(BASE_DIR)/paper

plot:
	git ls-files *.svg | sed s/svg// | xargs -I % inkscape $$PWD/%svg --export-pdf=%pdf -C
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 ../paper-figures.ipynb

pdf: bbl
	pdflatex snakemajoranas.tex
	pdflatex snakemajoranas.tex

bbl:
	pdflatex snakemajoranas.tex
	bibtex snakemajoranas.aux

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

paper:
	-mkdir -p $(BASE_DIR)
	-mkdir -p $(PAPER_DIR)
	cp snakemajoranas.bib $(PAPER_DIR)
	cp snakemajoranas.tex $(PAPER_DIR)
	export DATE=`date +%Y-%m-%d`
	sed -i '' -e "s/\\\today/`date +%Y-%m-%d`/g" $(PAPER_DIR)/snakemajoranas.tex
	-mkdir -p $(PAPER_DIR)/figures
	cp -r figures/*pdf $(PAPER_DIR)/figures/

data: code
	$(eval TARGET_DIR := $(BASE_DIR)/data_sub)
	-mkdir -p $(TARGET_DIR)
	-mkdir -p $(TARGET_DIR)/data
	cp -R ../data/* $(TARGET_DIR)/data
	-mv $(CODE_DIR)/* $(TARGET_DIR)
	cd $(TARGET_DIR) && zip -Zstore -r ../snakemajoranas_data.zip *

arxiv: paper bbl
	$(eval TARGET_DIR := $(BASE_DIR)/arxiv)
	-mkdir -p $(TARGET_DIR)
	cp snakemajoranas.bbl $(TARGET_DIR)
	-mv $(PAPER_DIR)/* $(TARGET_DIR)
	rm $(TARGET_DIR)/snakemajoranas.bib
	cd $(TARGET_DIR) && zip -r ../snakemajoranas_arxiv.zip *

journal: paper code
	$(eval TARGET_DIR := $(BASE_DIR)/journal)
	-mkdir -p $(TARGET_DIR)
	-mv $(CODE_DIR) $(TARGET_DIR)/supplementary
	-mv $(PAPER_DIR)/* $(TARGET_DIR)
	cd $(TARGET_DIR) && zip -r ../snakemajoranas_journal.zip *
