# Default target

DEVICE ?= 0
CONTINUE ?= 0 

.PHONY: all
all: opf cops

# Target that runs the Julia command to produce opf/results.jld2
opf: table-opf.tex
table-opf.tex: 
	@echo "Running Julia benchmarks to produce table-opf.tex..."
	julia --project --color=yes runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) opf
	julia --project --color=yes maketable.jl opf

# Target that runs the Julia command to produce opf/results.tex
miplib: table-miplib.tex
table-miplib.tex: 
	@echo "Running Julia benchmarks to produce table-miplib.tex..."
	julia --project --color=yes runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) miplib
	julia --project --color=yes maketable.jl miplib

# Target that runs the Julia command to produce opf/results.tex
cops: table-cops.tex
table-cops.tex: 
	@echo "Running Julia benchmarks to produce table-cops.tex..."
	julia --project --color=yes runbenchmark.jl --device $(DEVICE) $(if $(filter 1,$(CONTINUE)),--continue) cops
	julia --project --color=yes maketable.jl cops


# Optional clean target to remove the opf file
.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	rm -f *.tex

.PHONY: veryclean
veryclean: clean
	@echo "Cleaning up all generated files..."
	rm -f *.jld2 *.log
