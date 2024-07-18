
# Integrated Upstream Natural Gas and Electricity Planning 

This repository contains data, code, and result files for a research paper entitled: Integrating upstream natural gas and electricity planning in times of energy transition, authored by Majd Olleik, Hussein Tarhini, and Hans Auer.

## Table of Contents
1. [Description](#description)
2. [Project Structure](#project-structure)
3. [License](#license)

## Description

This project presents an integrated upstream natural gas and electricity planning optimization model that considers the upstream natural gas sector as a source of revenue to the state. The optimization model is a mixed integer non-linear program and is solved using a grid-search approach that solves a series of mixed integer linear programs to generate a globally optimal solution.

The model is applied to the case of Lebanon, a country in the Eastern Mediterranean region. A one-way and an extended sensitivity analyses on the main input parameters are performed. The data and results files are provided in their respective locations as further described in the [Project Structure](#project-structure) section.

<!--**Reference**: [Integrating upstream natural gas and electricity planning in times of energy transition](#)-->

Please cite this work as: Olleik, M., Tarhini, H., Auer, H., Integrated Upstream Natural Gas and Electricity Planning Mathematical Formulation, 2024.

## Project Structure
```markdown
├── Clustering/                            # Source codes and results for the generation of representative days
    ├── Data/                              # Data files used to generate the representative days       
├── Model/                                 # Model files
│   ├── main_v0_3_nothreading.py           # Main script
│   └── model_v0_6.py                      # Model formulation
│   └── input_file_v5.0.xlsx               # Input file for the model
├── Results/                               # Results and output files
│   └── Extended sensitivity scenarios/    # Output files for the sensitivity scenarios
│   └── Other sensitivities/               # Output files for some special sensitivities
│   └── Summary/                           # Processed summary files of the results
└── README.md                              # This README file
```
## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


