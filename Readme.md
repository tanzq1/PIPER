# PIPER: Pipeline-Aware Resource Provisioning for Serverless Query Processing

PIPER is a resource provisioning framework for serverless query processing. It extracts operator-level features from query execution plans and uses machine-learning models to predict execution time. Based on these predictions, it selects appropriate parallelism for individual stages, trying to minimize query latency and resource cost.

## Project Structure

```text
├── config/                       # Configuration management
│   ├── main_config.py            # Global configuration for datasets, methods, optimization algorithms, and defaults
│   ├── structure_config.py       # Operator types, feature definitions, and training-epoch configuration
│   └── mci_config.py             # Configuration for the MCI method
├── core/                         # Core data structures and runtime components
│   ├── plan_node.py              # Plan-tree nodes and ONNX inference wrapper
│   ├── onnx_manager.py           # ONNX model loading and management
│   ├── thread_block.py           # ThreadBlock abstraction and DOP selection logic
│   ├── pipeline_pair.py          # Pipeline-pair construction
│   └── padg_builder.py            # Pipeline Dependency Graph (PADG) construction
├── training/                     # Model training
│   ├── operator_dop_aware/       # DOP-aware operator models (PyTorch -> ONNX)
│   ├── operator_non_dop_aware/   # DOP-agnostic operator models (XGBoost/SVR -> ONNX)
│   ├── query_level/              # Query-level XGBoost models with Optuna hyperparameter tuning
│   ├── PPM/                      # PPM models (GNN / NN)
│   └── MCI/                      # MCI models (graph encoding + MLP)
├── inference/                    # Inference module
│   └── predict_queries.py        # Operator-level prediction and query-level time aggregation
├── optimization/                 # Optimization module
│   ├── optimizer.py              # Main Pipeline- and query-level DOP optimization workflow
│   ├── tree_updater.py           # Plan-tree DOP updates
│   ├── threading_utils.py        # Threading utilities
│   ├── result_processor.py       # Result export and multi-strategy comparison
│   └── moo_dop_optimizer.py      # NSGA-II multi-objective optimization (pymoo)
├── padg_moo/                      # PADG-MOO optimization algorithm
│   ├── algorithm.py              # Segment-wise competitive Top-K expansion and elite-pool iteration
│   ├── context.py                # Optimization context
│   ├── actions.py                # Action-space definition
│   ├── pool.py                   # Elite solution pool
│   ├── incremental_eval.py       # Incremental evaluation
│   └── run.py                    # PADG-MOO entry point
├── evaluation/                   # Evaluation module
│   └── evaluate_predictions.py   # Q-error aggregation and reporting
├── scripts/                      # Script entry points
│   ├── main.py                   # Main orchestration script (recommended)
│   ├── train.py                  # Training entry point (CLI)
│   ├── inference.py              # Inference entry point (CLI)
│   ├── evaluate.py               # Evaluation entry point (CLI)
│   ├── compare.py                # Strategy comparison (CLI)
│   └── ...
├── utils/                        # Utility functions
│   ├── dataset_loader.py         # Dataset loading and splitting
│   ├── path_utils.py             # Path management
│   ├── feature_engineering.py    # Feature engineering
│   └── logger.py                 # Logging
├── data_kunpeng/                 # Experimental data (TPC-H / TPC-DS / JOB)
├── input/                        # Baseline data for comparison
├── output/                       # Models, prediction results, and optimization results
└── docs/                         # Supplementary documentation
```

## Functionality

### 1. Model Training

The project supports the following prediction models, with ONNX used to standardize the inference interface:

| Model | Method | Description |
| --- | --- | --- |
| DOP-aware operator model | PyTorch MLP -> ONNX | Trained separately for each operator type; the input includes DOP. |
| DOP-agnostic operator model | XGBoost / SVR -> ONNX | Trained separately for each operator type; the input excludes DOP. |
| Query-level model | XGBoost + Optuna -> ONNX | Predicts total execution time from query-level features. |
| PPM | GNN / NN -> ONNX | Fits DOP-latency and DOP-memory relationship curves. |
| MCI | Graph encoding (GAT) + MLP | Predicts latency at different DOP values at the Pipeline granularity. |

The DOP-aware and DOP-agnostic operator models are trained in this project. The query-level model, PPM, and MCI are methods from other work.

### 2. Inference

The inference workflow reads test-set operator information from `plan_info.csv`, loads trained ONNX models, predicts execution time and memory consumption for each operator, and aggregates the predictions into query-level results according to the plan-tree structure.

### 3. Pipeline-Aware Resource Provisioning

After obtaining operator-level predictions, PIPER provisions parallel resources by selecting DOP values for each Pipeline/ThreadBlock in a query plan. The following optimization strategies are supported:

- **Pipeline optimization:** selects DOP for each ThreadBlock top-down using heuristic rules and flow-rate interval matching.
- **Query-level optimization:** directly predicts total execution time under different DOP values using a query-level model.
- **NSGA-II multi-objective optimization:** jointly optimizes latency, resource cost (`DOP x latency`), and flow-rate matching.
- **PADG-MOO:** segment-wise competitive multi-objective optimization based on the Pipeline Dependency Graph.

### 4. Evaluation and Comparison

Prediction accuracy is evaluated using Q-error. The project also supports performance comparisons with multiple baseline strategies, including default DOP, Auto-DOP, PPM, and FGO.

## Dependencies

```text
python >= 3.8
numpy
pandas
torch
onnxruntime
xgboost
optuna
scikit-learn
skl2onnx
onnxmltools
pymoo          # Optional; used for NSGA-II multi-objective optimization
```

## Quick Start

### Data Preparation

Place experimental data under `data_kunpeng/`. The directory structure must conform to the `DATASETS` definition in `config/main_config.py`. Each dataset must contain `plan_info.csv` (operator-level information) and `query_info.csv` (query-level information).

### Option 1: Main Orchestration Script (Recommended)

Edit the configuration section in `scripts/main.py` to set the dataset, training method, optimization algorithm, and related parameters, then run:

```bash
python scripts/main.py
```

The following variables control the workflow:

```python
DATASET = 'tpcds'                    # Dataset: 'tpch', 'tpcds', 'job'
TRAIN_MODE = 'estimated_train'       # Training mode: 'exact_train', 'estimated_train'
TRAIN_METHOD = 'dop_aware'           # Training method: 'dop_aware', 'non_dop_aware', 'ppm', 'query_level', 'mci'
OPTIMIZATION_ALGORITHM = 'pipeline'  # Optimization algorithm: 'pipeline', 'query_level', 'auto_dop', 'ppm'

RUN_TRAIN = True       # Run training
RUN_INFERENCE = True   # Run inference
RUN_OPTIMIZE = True    # Run optimization
RUN_EVALUATE = True    # Run evaluation
RUN_COMPARE = True     # Run strategy comparison
```

### Option 2: Standalone Scripts

```bash
# Training
python scripts/train.py --method dop_aware --dataset tpcds --train_mode estimated_train

# Inference
python scripts/inference.py --dataset tpcds --train_mode estimated_train --use_estimates

# Evaluation
python scripts/evaluate.py --dataset tpcds --train_mode estimated_train

# Strategy comparison
python scripts/compare.py --dataset tpcds
```

## Output Directory Structure

```text
output/<dataset>/
├── models/
│   └── <train_mode>/
│       ├── operator_dop_aware/       # DOP-aware models (.onnx)
│       ├── operator_non_dop_aware/   # DOP-agnostic models (.onnx)
│       ├── query_level/              # Query-level models (.onnx)
│       └── PPM/                      # PPM models
├── predictions/
│   ├── operator_level/               # Operator-level prediction results
│   ├── query_level/                  # Query-level prediction results
│   └── PPM/
├── evaluations/
│   └── qerror_summary_report.csv     # Q-error evaluation report
└── optimization_results/
    ├── query_details_optimized.json  # Optimized query details
    ├── operators_optimized.csv       # Operator-level optimization results
    ├── query_max_dop_optimized.csv   # Maximum DOP per query
    └── timing_log.csv                # Optimization timing log
```

## Supported Datasets

| Dataset | Training directory | Test directory |
| --- | --- | --- |
| TPC-H | `tpch_output_500` | `tpch_output_22` |
| TPC-DS | `tpcds_100g_output_train` | `tpcds_100g_output_test` |
| JOB | `job_100g_data_train` | `job_100g_data_test` |

## Supported Operator Types

The system models more than 20 major operators in the openGauss vectorized execution engine, including CStore Scan, Vector Hash Join, Vector Sonic Hash Join, Vector Sort, Vector Hash Aggregate, and Vector Streaming (`LOCAL GATHER` / `REDISTRIBUTE` / `BROADCAST`). See `config/structure_config.py` for the complete list.

## Supplementary Documentation

- [MOO Pipeline Optimization](docs/MOO_PIPELINE_OPTIMIZATION.md)
- [Inference Method Comparison](docs/inference_methods_comparison.md)
- [Script Usage Guide](scripts/README.md)
