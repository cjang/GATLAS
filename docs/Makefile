# This requires Graphviz
# http://www.graphviz.org/

# GraphViz diagram drawing tool
DOT = dot

# preprocessor for GraphViz dot
DPP = ./dotpp

# generate network diagrams using GraphViz
.SUFFIXES : .dpp .svg
.dpp.svg :
	$(DPP) $< | $(DOT) -Tsvg > $@

DIAGRAMS = \
	matmulOverview.svg

all : $(DIAGRAMS)

clean :
	rm -f *.svg

