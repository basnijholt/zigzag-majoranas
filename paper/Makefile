BASE_DIR = ../tmp
CODE_DIR = $(BASE_DIR)/code
PAPER_DIR = $(BASE_DIR)/paper

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
	cp ../environment.yaml $(CODE_DIR)/
	cp ../YOUR_NOTEBOOK.ipynb $(CODE_DIR)/
	cp ../THE_FUNCTIONS.py $(CODE_DIR)/

paper:
	-mkdir -p $(BASE_DIR)
	-mkdir -p $(PAPER_DIR)
	cp snakemajoranas.bib $(PAPER_DIR)
	cp snakemajoranas.tex $(PAPER_DIR)
	export DATE=`date +%Y-%m-%d`
	sed -i '' -e "s/\\\today/`date +%Y-%m-%d`/g" $(PAPER_DIR)/snakemajoranas.tex
	cp -r figures $(PAPER_DIR)

data: code
	$(eval TARGET_DIR := $(BASE_DIR)/data_sub)
	-mkdir -p $(TARGET_DIR)
	-mkdir -p $(TARGET_DIR)/data
	cp ../data/*.h5 $(TARGET_DIR)/data
	-mv $(CODE_DIR)/LICENSE.txt $(TARGET_DIR)
	-mv $(CODE_DIR)/README.txt $(TARGET_DIR)
	-mv $(CODE_DIR) $(TARGET_DIR)
	cp ../DATA_README.md $(TARGET_DIR)/DATA_README.txt
	cd $(TARGET_DIR) && zip -Zstore -r ../snakemajoranas_data.zip *

arxiv: paper bbl code
	$(eval TARGET_DIR := $(BASE_DIR)/arxiv)
	-mkdir -p $(TARGET_DIR)
	-mkdir -p $(TARGET_DIR)/anc/
	-mv $(CODE_DIR)/* $(TARGET_DIR)/anc/
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
