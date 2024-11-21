# Flexible Endpoint Titer Analysis Tool

This tool processes ELISA data to calculate antibody titers using optimized logistic regression (4-parameter and 5-parameter logistic models). The tool is designed to be highly flexible, supporting various data structures and formats while maintaining high accuracy.

## Features

* **Flexible data structure support**
  - Variable number of dilution columns
  - Continuous or discontinuous data blocks
  - Dynamic sample block detection
* **Advanced fitting capabilities**
  - 4-parameter and 5-parameter logistic regression
  - Automatic model selection based on AIC
  - Optimized initial parameter estimation
* **Comprehensive analysis metrics**
  - Statistical evaluation (R², Adjusted R², AIC, BIC, RMSE)
  - Detailed fitting quality assessment
  - Verbose logging option for analysis tracking
* **Enhanced visualization**
  - High-resolution plot generation
  - Excel integration with embedded plots
  - Individual PNG exports for each sample

## Requirements

* Python 3.8+
* Required packages: `openpyxl`, `pandas`, `numpy`, `matplotlib`, `scipy`

## Installation

Clone the repository and install the required packages:

```bash
# Clone the repository
git clone https://github.com/dacchi-s/flexible-endpoint-titer.git

# Change to the project directory
cd flexible-endpoint-titer

# Install required packages using pip
pip install -r requirements.txt
```

Alternatively, for conda users:

```bash
# Clone the repository
git clone https://github.com/dacchi-s/flexible-endpoint-titer.git

# Change to the project directory
cd flexible-endpoint-titer

# Create and activate conda environment
conda env create -f flexible-endpoint-titer.yml
conda activate flexible-endpoint-titer-analysis
```

## help
```
usage: flexible-endpoint-titer.py [-h] --input INPUT --cutoff CUTOFF [--method {4,5,auto}] [--replicates {1,2}] [--verbose]

ELISA Endpoint Titer Analysis Tool - Flexible Data Structure Version

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input Excel file
  --cutoff CUTOFF, -c CUTOFF
                        Cutoff value
  --method {4,5,auto}, -m {4,5,auto}
                        Fitting method (4: 4PL, 5: 5PL, auto: automatic selection) Default: auto
  --replicates {1,2}, -r {1,2}
                        Number of technical replicates (1: single, 2: duplicate) Default: 2
  --verbose, -v        Display detailed output

Usage Examples:
  Basic usage:
    flexible-endpoint-titer.py -i "path/to/data.xlsx" -c 0.2
  
  Single data analysis:
    flexible-endpoint-titer.py -i "path/to/single_data.xlsx" -c 0.15 -r 1
  
  Specify 4PL fitting:
    flexible-endpoint-titer.py -i "path/to/data.xlsx" -c 0.2 -m 4
  
  Output detailed analysis information:
    flexible-endpoint-titer.py -i "path/to/data.xlsx" -c 0.2 -v

Input File Format:
  - Excel format (.xlsx)
  - Place measurement data in Sheet1
  - Row 1: Dilution rates
  - Row 2 and beyond: Sample names and measured values
```

## Usage

Basic usage:
```bash
python flexible-endpoint-titer.py --input path/to/input.xlsx --cutoff 0.2
```

Advanced options:
```bash
python flexible-endpoint-titer.py --input path/to/input.xlsx --cutoff 0.2 --method <4|5|auto> --replicates <1|2> --verbose
```

### Example Usage Scenarios

1. Basic analysis with default settings:
```bash
python flexible-endpoint-titer.py --input example_data.xlsx --cutoff 0.1
```

2. Single replicate analysis:
```bash
python flexible-endpoint-titer.py --input single_data.xlsx --cutoff 0.15 --replicates 1
```

3. Forced 4PL fitting with detailed output:
```bash
python flexible-endpoint-titer.py --input data.xlsx --cutoff 0.2 --method 4 --verbose
```

## Input Data Format

The tool supports various data formats:

### Pattern 1: Continuous Data
dilution   100    200    400    800   1600   3200
sample-1   0.045  0.028  0.024  0.014  0.016  0.013
sample-1   0.045  0.027  0.019  0.016  0.015  0.014
sample-2   0.097  0.047  0.029  0.018  0.013  0.012
sample-2   0.099  0.047  0.028  0.018  0.014  0.011

### Pattern 2: With Empty Lines
dilution   100    200    400    800   1600   3200
sample-1   0.045  0.028  0.024  0.014  0.016  0.013
sample-1   0.045  0.027  0.019  0.016  0.015  0.014

sample-2   0.097  0.047  0.029  0.018  0.013  0.012
sample-2   0.099  0.047  0.028  0.018  0.014  0.011

### Pattern 3: Different Block Sizes
dilution   100    200    400    800
sample-1   0.045  0.028  0.024  0.014
sample-1   0.045  0.027  0.019  0.016
sample-2   0.097  0.047  0.029  0.018
sample-2   0.099  0.047  0.028  0.018

## Outputs

1. **Analysis Results** (`results_<input_filename>.xlsx`)
   - Results sheet: Numerical data (titers, statistics)
   - Plots sheet: Embedded graphs for each sample

2. **Individual Plots** (`plots/<sample_name>_plot.png`)
   - High-resolution plots for each sample
   - Log-scale dilution axis
   - Fitted curves with cutoff indicators

3. **Analysis Log** (`analysis_log_<input_filename>.txt`, when using --verbose)
   - Detailed processing information
   - Fitting statistics and warnings
   - Error reports if any

## Notes

- Supports both Windows and macOS Japanese fonts
- Automatically detects and processes dilution rates
- Handles Excel formulas in dilution rate cells
- Memory usage scales with input data size
