# Serverless Predictor Usage Guide (After Refactoring)

## 🎯 Refactoring Goals
- **Integrate similar functionality**: Consolidate multiple training/inference/optimization scripts into unified entry points
- **Reduce code duplication**: Extract common configuration and utility functions
- **Clearer code structure**: Control different methods through parameters

## 📁 New Script Structure

```
scripts/
├── config.py          # Unified configuration management
├── utils.py           # Common utility functions
├── main.py            # Main control script (recommended)
├── train.py           # Unified training entry point
├── inference.py       # Unified inference entry point
├── optimize.py        # Unified optimization entry point
├── evaluate.py        # Unified evaluation entry point
└── compare.py         # Comparison analysis
```

## 🚀 Usage

### Method 1: Using Main Control Script (Recommended)

Run `main.py` directly, control parameters by modifying variables in the file:

```bash
python main.py
```

Modify the configuration area in `main.py`:

```python
# ==================== Configuration Area ====================
# Basic configuration
DATASET = 'tpcds'  # Dataset: 'tpch' or 'tpcds'
TRAIN_MODE = 'estimated_train'  # Training mode: 'exact_train' or 'estimated_train'
USE_ESTIMATES_MODE = True  # Whether to use estimates

# Training configuration
TRAIN_METHOD = 'dop_aware'  # Training method: 'dop_aware', 'non_dop_aware', 'ppm', 'query_level'
PPM_TYPE = 'GNN'  # PPM type: 'GNN' or 'NN'

# Optimization configuration
OPTIMIZATION_ALGORITHM = 'pipeline'  # Optimization algorithm: 'pipeline', 'query_level', 'auto_dop', 'ppm'

# Runtime control - set which functions to run (True/False)
RUN_TRAIN = True  # Whether to run training
RUN_INFERENCE = True  # Whether to run inference
RUN_OPTIMIZE = True  # Whether to run optimization
RUN_EVALUATE = True  # Whether to run evaluation
RUN_COMPARE = True  # Whether to run comparison analysis
# =======================================================
```

### Method 2: Using Individual Scripts Directly

```bash
# Training
python train.py --method dop_aware --dataset tpcds --train_mode estimated_train

# Inference
python inference.py --dataset tpcds --train_mode estimated_train --use_estimates

# Optimization
python optimize.py --algorithm pipeline --dataset tpcds --train_mode estimated_train

# Evaluation
python evaluate.py --dataset tpcds --train_mode estimated_train

# Comparison
python compare.py --dataset tpcds
```

## ⚙️ Configuration

### Dataset Configuration
- `tpch`: TPC-H dataset
- `tpcds`: TPC-DS dataset

### Training Methods
- `dop_aware`: DOP-aware operator models/ pipeline
- `non_dop_aware`: Non-DOP-aware operator models
- `ppm`: PPM method (requires specifying --ppm_type GNN/NN)
- `query_level`: Query-level models
- `mci`: stage independent models

### Training Modes
- `exact_train`: Exact training
- `estimated_train`: Estimated training

### Optimization Algorithms
- `pipeline`: Pipeline optimization (core method)
- `query_level`: Query-level optimization
- `auto_dop`: Auto-DOP method
- `ppm`: PPM method
- `mci`: stage independent method

## 🔧 Parameter Description

### Training Parameters
- `--method`: Training method (required)
- `--dataset`: Dataset name (default: tpcds)
- `--train_mode`: Training mode (default: estimated_train)
- `--ppm_type`: PPM method type (default: GNN)
- `--total_queries`: Total number of queries (default: 500)
- `--train_ratio`: Training ratio (default: 1.0)
- `--n_trials`: XGBoost optimization trial count (default: 30)

### Inference Parameters
- `--dataset`: Dataset name (default: tpcds)
- `--train_mode`: Training mode (default: estimated_train)
- `--use_estimates`: Whether to use estimates (default: True)

### Optimization Parameters
- `--algorithm`: Optimization algorithm (required)
- `--dataset`: Dataset name (default: tpcds)
- `--train_mode`: Training mode (default: estimated_train)
- `--base_dop`: Baseline DOP (default: 64)
- `--min_improvement_ratio`: Minimum improvement ratio (default: 0.2)
- `--min_reduction_threshold`: Minimum reduction threshold (default: 200)
- `--use_estimates`: Whether to use estimates (default: True)

## 📊 Output Structure

```
output/
├── tpcds/
│   ├── models/
│   │   ├── estimated_train/
│   │   │   ├── operator_dop_aware/
│   │   │   ├── operator_non_dop_aware/
│   │   │   ├── query_level/
│   │   │   └── PPM/
│   ├── predictions/
│   │   ├── operator_level/
│   │   ├── query_level/
│   │   └── PPM/
│   ├── evaluations/
│   │   └── qerror_summary_report.csv
│   └── optimization_results/
│       ├── query_details_optimized.json
│       ├── operators_optimized.csv
│       ├── query_max_dop_optimized.csv
│       └── timing_log.csv
└── evaluations/
    └── optimization_comparison_report.csv
```

## 🔄 Workflow Examples

### Complete Experiment Workflow
```bash
# 1. Train all models
python main.py  # Set TRAIN_METHOD to 'dop_aware', set RUN_TRAIN = True
python main.py  # Set TRAIN_METHOD to 'non_dop_aware', set RUN_TRAIN = True
python main.py  # Set TRAIN_METHOD to 'ppm', set RUN_TRAIN = True
python main.py  # Set TRAIN_METHOD to 'query_level', set RUN_TRAIN = True

# 2. Run inference
python main.py  # Set RUN_INFERENCE = True

# 3. Run optimization
python main.py  # Set RUN_OPTIMIZE = True

# 4. Run evaluation
python main.py  # Set RUN_EVALUATE = True

# 5. Run comparison analysis
python main.py  # Set RUN_COMPARE = True
```

## 🆚 Comparison with Original Scripts

### Original Approach (11 independent scripts)
```bash
python run_dop_aware_training.py
python run_non_dop_aware_training.py
python run_PPM_train.py
python run_query_level_training.py
python run_inference.py
python run_optimization.py
python run_query_level_optimizer.py
python run_evaluation.py
python run_comparison.py
python run_create_datasplit.py
python run_consolidated_timing_analysis.py
```

### Refactored Approach (1 main control script)
```bash
python main.py  # Control runtime parameters by modifying variables in the file
```

## ✅ Refactoring Benefits

1. **Code reuse**: Eliminates duplicate configuration and path setup code
2. **Unified interface**: All functionality controlled through one main script
3. **Parameterization**: Control different methods by modifying variables in the file, no command-line arguments needed
4. **Easy maintenance**: Centralized configuration management, easier to modify
5. **Error handling**: Unified error handling and logging
6. **Complete documentation**: Each script has detailed help information

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Ensure project root directory is in Python path
2. **File not found**: Check if data file paths are correct
3. **Model not found**: Ensure corresponding models have been trained
4. **Permission issues**: Ensure write permissions for output directory

### Debug Mode
```bash
# Run main script directly
python main.py

# Or run individual function scripts
python train.py --help
python optimize.py --help
```
