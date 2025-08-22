# Default target

DEVICE ?= 0
CONTINUE ?= 0

.PHONY: all
all: opf cops miplib


miplib: results/table-miplib.tex results/fulltable-miplib-high.txt results/fulltable-miplib-low.txt
results/table-miplib.tex: 
	@echo "Running Julia benchmarks to produce table-miplib.tex..."
	julia --project=lp --color=yes lp/runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue)
results/fulltable-miplib-high.txt fulltable-miplib-low.txt:
	julia --project=lp --color=yes lp/maketable.jl
	julia --project=lp --color=yes lp/fulltable.jl

opf: results/table-opf.tex results/fulltable-opf-high.txt results/fulltable-opf-low.txt
results/table-opf.tex: 
	@echo "Running Julia benchmarks to produce table-opf.tex..."
	julia --project=nlp --color=yes nlp/runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) opf
results/fulltable-opf-high.txt results/fulltable-opf-low.txt:
	julia --project=nlp --color=yes nlp/maketable.jl opf
	julia --project=nlp --color=yes nlp/fulltable.jl opf

cops: results/table-cops.tex results/fulltable-cops-high.txt results/fulltable-cops-low.txt
results/table-cops.tex: 
	@echo "Running Julia benchmarks to produce table-cops.tex..."
	julia --project=nlp --color=yes nlp/runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) cops
results/fulltable-cops-high.txt results/fulltable-cops-low.txt:
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
