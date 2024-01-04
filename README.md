# cardinality estimation based on ML

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

#### 2.2.1 single table query

```
python run.py --evaluate --model models/model_stats_gaussian_1000_cdf.pkl --query workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql
```

### 2.2.2 sub-plan query

# Tests

```
python -m unittest
```

```
https://github.com/VincentStimper/normalizing-flows
https://github.com/tommyod/KDEpy
```

## TODO

0. fast interp with numba
1. kernel = box or gaussian
2. coutour plot log
3. 1d bw selection. lead to negative density. (Interpolate error! select the most appropriate one!) (a) manual bw to 500. (b), granularity to 2\*\*12. how to auto decide it?
4. 2d density could not be too small. if p<1e-5, division on small probabilities gives rise to higher error. [this could be explained as follows. the cases with multiple selections in a single table, it is not a good solution to have grid>1000, it is because many prababilities are tiny numbers, lead to division peak!]
5. 2d dropnan affect single table estimation significantly.

# shortcuts

```
docker run -p 5432:5432 -d ceb1
```
