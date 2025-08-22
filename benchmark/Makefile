# Default target

DEVICE ?= 0
CONTINUE ?= 0

.PHONY: all
all: opf cops miplib


miplib: table-miplib.tex fulltable-miplib-high.txt fulltable-miplib-low.txt
table-miplib.tex: 
	@echo "Running Julia benchmarks to produce table-miplib.tex..."
	julia --project=lp --color=yes lp/runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) miplib
	julia --project=lp --color=yes lp/maketable.jl miplib
	julia --project=lp --color=yes lp/fulltable.jl cops

opf: table-opf.tex fulltable-miplib-high.txt fulltable-miplib-low.txt
table-opf.tex: 
	@echo "Running Julia benchmarks to produce table-opf.tex..."
	julia --project=nlp --color=yes nlp/runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) opf
	julia --project=nlp --color=yes nlp/maketable.jl opf
	julia --project=nlp --color=yes nlp/fulltable.jl cops

cops: table-cops.tex fulltable-miplib-high.txt fulltable-miplib-low.txt
table-cops.tex: 
	@echo "Running Julia benchmarks to produce table-cops.tex..."
	julia --project=nlp --color=yes nlp/runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) cops
	julia --project=nlp --color=yes nlp/maketable.jl cops
	julia --project=nlp --color=yes nlp/fulltable.jl cops


.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	rm -f nlp/*.tex nlp/*.txt lp/*.tex lp/*.txt

.PHONY: veryclean
veryclean: clean
	@echo "Cleaning up all generated files..."
	rm -f nlp/*.jld2 nlp/*.log lp/*.jld2 lp/*.log
