# GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers

## Overview
This repository contains the source code and materials for the paper titled "GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers." The paper presents an overview of GPU-accelerated optimization solvers based on second-order methods, emphasizing pivoting-free interior-point methods for large and sparse linear and nonlinear programs.

## Authors
- Alexis Montoison, Argonne National Laboratory
- François Pacaud, Mines Paris-PSL
- Sungho Shin, Massachusetts Institute of Technology
- Mihai Anitescu, Argonne National Laboratory

## Benchmark
### Dependencies
To reproduce the results in this paper, the following software or license file should be obtained:
- [julia](https://julialang.org/downloads/): we recommend using [juliaup](https://github.com/JuliaLang/juliaup) to install the latest stable version of Julia.
- [libHSL](https://licences.stfc.ac.uk/product/libhsl-2025_7_21): a library for sparse linear algebra. After downloading libHSL, please install `HSL_jll` using the locally downloaded library.
```shell
julia -e 'using Pkg; Pkg.develop(path="path/to/libhsl")'
```
- [Gurobi](https://www.gurobi.com): A free academic license can be obtained from Gurobi's website. After obtaining the license, `gurobi.lic` should be placed in the home directory.

### Running Benchmarks
```
make -C benchmark
```
