# cardinatlity estimation based on ML

## 1. Environment Setup

```
conda create -n cardi python=3.11
```

## 2. STATS

### 2.1 pre-process the data

first, convert timestamps to integers by

```
python run.py --dataset stats --preprocess
--data_folder End-to-End-CardEst-Benchmark/datasets/stats_simplified/
```

Note, this command will override the existing file. thus, you could make a copy of this dataset before pre-processing it.

### 2.1 train model

```
python run.py --dataset stats \
              --train
```

### 2.2 evaluate
