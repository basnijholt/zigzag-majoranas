BASE_DIR = ../tmp
CODE_DIR = $(BASE_DIR)/code
PAPER_DIR = $(BASE_DIR)/paper

plot:
	git ls-files *.svg | sed s/svg// | xargs -I % inkscape $$PWD/%svg --export-pdf=%pdf -D
	jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 ../paper-figures.ipynb

pdf: bbl
	pdflatex report.tex
	pdflatex report.tex

bbl:
	pdflatex report.tex
	bibtex report.aux

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
	cp report.bib $(PAPER_DIR)
	cp report.tex $(PAPER_DIR)
	export DATE=`date +%Y-%m-%d`
	sed -i '' -e "s/\\\today/`date +%Y-%m-%d`/g" $(PAPER_DIR)/report.tex
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
	cd $(TARGET_DIR) && zip -Zstore -r ../report_data.zip *
