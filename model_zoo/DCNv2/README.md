## DCNv2 训练说明

本文档说明如何在当前目录下完整复现一套 DCNv2 训练流程。
覆盖两种场景：

1. 你已经有真实业务数据，需要先转换成 FuxiCTR 需要的 wide CSV 格式。
2. 你暂时没有真实数据，想先用模拟数据把整条训练链路跑通。

下面所有命令都默认在仓库根目录执行。

### 1. 环境准备

先安装 Python 依赖：

```bash
pip install -r requirements.txt
```

如果使用 GPU 训练，还需要确保当前 PyTorch 安装与 CUDA 环境匹配。

### 2. 训练入口

DCNv2 的训练入口如下：

```bash
PYTHONPATH=. python -m model_zoo.DCNv2.run_expid --config model_zoo/DCNv2/config --expid <EXP_ID> --device <DEVICE>
```

常见的 `device` 取值：

```bash
--device cpu
--device cuda:0
--device npu:0
```

### 3. 方案 A：使用真实数据训练

如果你的原始数据仍然是稀疏的 NPS 风格格式，需要先转换再训练，使用这一套流程。

#### 3.1 准备原始输入文件

转换脚本需要两类输入：

1. 一份原始 CSV，每行代表一个用户。
2. 一份 mapping CSV，用来描述每个 sparse key 的含义。

如果你当前没有现成的 `mapping_csv`，而是只有一个 schema JSON，可以先用
[model_zoo/DCNv2/generate_mapping_csv_from_schema_json.py](d:/code/FuxiCTR/model_zoo/DCNv2/generate_mapping_csv_from_schema_json.py)
生成 mapping CSV。

例如 schema JSON 如下：

```json
{
	"value_size_list": [-1, 3],
	"key_size": 253,
	"key_parts": ["age", "gender"],
	"time_size": 6
}
```

转换规则如下：

1. `key` 按 `0` 到 `key_size - 1` 顺序生成。
2. `feature_name` 按 `key_parts` 顺序循环取值。
3. `feature_type` 按 `value_size_list` 顺序循环取值。
4. `value_size_list` 中如果值为 `-1`，则映射成 `numeric`；否则映射成 `categorical`。

生成命令：

```bash
PYTHONPATH=. python model_zoo/DCNv2/generate_mapping_csv_from_schema_json.py \
	--schema_json <SCHEMA_JSON> \
	--output_csv <MAPPING_CSV>
```

原始 CSV 至少要包含这些列：

```text
user_id, key, time, value, label
```

可选列：

```text
rating
```

字段含义：

1. `user_id`: 实体主键。
2. `key`: 逗号分隔的 sparse feature key 列表。
3. `time`: 逗号分隔的时间片索引列表。
4. `value`: 与 `key`、`time` 对齐的逗号分隔取值列表。
5. `label`: 训练标签。
6. `rating`: 可选，每个 period 对应的评分列表。

mapping CSV 至少要包含：

```text
key, feature_type
```

可选列：

```text
feature_name
```

`feature_type` 最终会被归一成 `numeric` 或 `categorical`。

#### 3.2 将原始数据转换成 Wide CSV

使用 [model_zoo/DCNv2/convert_nps_csv_for_dcnv2.py](d:/code/FuxiCTR/model_zoo/DCNv2/convert_nps_csv_for_dcnv2.py)：

```bash
PYTHONPATH=. python model_zoo/DCNv2/convert_nps_csv_for_dcnv2.py \
	--input_csv <RAW_INPUT_CSV> \
	--mapping_csv <MAPPING_CSV> \
	--output_csv data/nps_wide/wide.csv \
	--schema_json data/nps_wide/schema.json \
	--periods 6 \
	--user_col user_id \
	--key_col key \
	--time_col time \
	--value_col value \
	--rating_col rating \
	--label_col label \
	--mapping_key_col key \
	--mapping_type_col feature_type \
	--mapping_name_col feature_name \
	--strict
```

如果你的原始 CSV 里没有 `rating` 列，就不要强行传一个不存在的列名；只有真实存在时才传对应列名。

#### 3.3 将 Wide CSV 切分成 Train/Test

使用 [model_zoo/DCNv2/split_wide_csv.py](d:/code/FuxiCTR/model_zoo/DCNv2/split_wide_csv.py)：

```bash
PYTHONPATH=. python model_zoo/DCNv2/split_wide_csv.py \
	--input_csv data/nps_wide/wide.csv \
	--train_csv data/nps_wide/train.csv \
	--test_csv data/nps_wide/test.csv \
	--test_ratio 0.3 \
	--label_col label
```

当前这套流程里，`test.csv` 同时作为验证集和测试集使用。

#### 3.4 生成 Dataset 配置片段

使用 [model_zoo/DCNv2/generate_dataset_config_from_schema.py](d:/code/FuxiCTR/model_zoo/DCNv2/generate_dataset_config_from_schema.py)：

```bash
PYTHONPATH=. python model_zoo/DCNv2/generate_dataset_config_from_schema.py \
	--schema_json data/nps_wide/schema.json \
	--dataset_id nps_wide_csv \
	--data_root ../../data/ \
	--train_data ../../data/nps_wide/train.csv \
	--valid_data ../../data/nps_wide/test.csv \
	--test_data ../../data/nps_wide/test.csv
```

把命令输出的 YAML 片段复制到 [model_zoo/DCNv2/config/dataset_config.yaml](d:/code/FuxiCTR/model_zoo/DCNv2/config/dataset_config.yaml) 中。

如果你的任务是多分类，最终的 dataset 配置要确保是：

```yaml
label_col: {name: label, dtype: int}
```

如果你的任务是二分类，保留 `float` 类型也可以。

#### 3.5 准备 Model 配置

编辑 [model_zoo/DCNv2/config/model_config.yaml](d:/code/FuxiCTR/model_zoo/DCNv2/config/model_config.yaml)。

如果是二分类，可以从 `DCNv2_nps_binary` 开始改。

重点参数如下：

1. `dataset_id`: 必须与 `dataset_config.yaml` 里的数据块名称一致。
2. `task`: `binary_classification` 或 `multiclass_classification`。
3. `loss`: 二分类用 `binary_crossentropy`，多分类用 `cross_entropy`。
4. `num_classes`: 多分类必填，并且要与真实标签类别数一致。
5. `metrics`: 例如二分类可用 `['logloss', 'AUC']`，多分类可用 `['accuracy', 'logloss', 'AUC']`。
6. `monitor`: early stop 和最佳模型保存所监控的指标，例如 `AUC` 或 `accuracy`。
7. `batch_size`、`learning_rate`、`embedding_dim`、`epochs`: 主要训练超参数。
8. `topk_metrics`: 可选，例如 `[100, 500]`。
9. `topk_output_dir`: 可选，用于输出 topK CSV 报表。
10. `topk_analysis_cols`: 可选，例如 `['raw_rating']`，用于在 topK 排序结果中额外统计指定列的分布。

多分类的最小配置模式如下：

```yaml
task: multiclass_classification
num_classes: 3
loss: 'cross_entropy'
metrics: ['accuracy', 'logloss', 'AUC']
monitor: 'accuracy'
monitor_mode: 'max'
```

#### 3.6 启动训练

二分类示例：

```bash
PYTHONPATH=. python -m model_zoo.DCNv2.run_expid --config model_zoo/DCNv2/config --expid DCNv2_nps_binary --device cuda:0
```

多分类示例：

```bash
PYTHONPATH=. python -m model_zoo.DCNv2.run_expid --config model_zoo/DCNv2/config --expid DCNv2_nps_multiclass --device cuda:0
```

### 4. 方案 B：没有真实数据时的训练流程

如果你想先验证整条链路，而不是立刻接入真实数据，使用这套流程。

#### 4.1 二分类 Mock 数据

使用 [model_zoo/DCNv2/generate_mock_nps_wide_data.py](d:/code/FuxiCTR/model_zoo/DCNv2/generate_mock_nps_wide_data.py)：

```bash
PYTHONPATH=. python model_zoo/DCNv2/generate_mock_nps_wide_data.py --output_dir data/nps_wide --num_samples 10000
```

然后切分数据：

```bash
PYTHONPATH=. python model_zoo/DCNv2/split_wide_csv.py \
	--input_csv data/nps_wide/wide.csv \
	--train_csv data/nps_wide/train.csv \
	--test_csv data/nps_wide/test.csv \
	--test_ratio 0.3 \
	--label_col label
```

然后训练：

```bash
PYTHONPATH=. python -m model_zoo.DCNv2.run_expid --config model_zoo/DCNv2/config --expid DCNv2_nps_binary --device cuda:0
```

#### 4.2 多分类 Mock 数据

使用 [model_zoo/DCNv2/generate_mock_nps_wide_data_multiclass.py](d:/code/FuxiCTR/model_zoo/DCNv2/generate_mock_nps_wide_data_multiclass.py)：

```bash
PYTHONPATH=. python model_zoo/DCNv2/generate_mock_nps_wide_data_multiclass.py \
	--output_dir data/nps_wide_multiclass \
	--num_samples 10000 \
	--num_classes 3
```

这份 mock 数据除了训练用的 `label`，还会额外保留 `raw_rating` 列。
其中 `label` 是由 `raw_rating` 按区间分组后得到的多分类标签，`raw_rating` 本身不会参与模型计算，默认仅用于 topK 分析导出。

然后切分数据：

```bash
PYTHONPATH=. python model_zoo/DCNv2/split_wide_csv.py \
	--input_csv data/nps_wide_multiclass/wide.csv \
	--train_csv data/nps_wide_multiclass/train.csv \
	--test_csv data/nps_wide_multiclass/test.csv \
	--test_ratio 0.3 \
	--label_col label
```

然后训练：

```bash
PYTHONPATH=. python -m model_zoo.DCNv2.run_expid --config model_zoo/DCNv2/config --expid DCNv2_nps_multiclass --device cuda:0
```

### 5. 需要重点关注的参数

#### 5.1 数据参数

1. `periods`: 必须与真实时间片数量一致。
2. `label_col`: 必须指向真实标签列名。
3. `label_col.dtype`: 二分类用 `float`，多分类用 `int`。
4. `mapping_type_col`: 必须正确区分 numeric 和 categorical 特征。
5. `min_categr_count`: 用于控制低频类别过滤。

#### 5.2 训练参数

1. `task`: 决定模型走二分类还是多分类逻辑。
2. `num_classes`: 多分类必填，并且要与实际类别数一致。
3. `loss`: 二分类用 `binary_crossentropy`，多分类用 `cross_entropy`。
4. `metrics`: 选择与任务匹配的评测指标。
5. `monitor` 和 `monitor_mode`: 控制 checkpoint 保存和 early stop 的依据。
6. `device`: 根据环境选择 `cpu`、`cuda:0` 或 `npu:0`。

#### 5.3 TopK 分析参数

如果你在 model 配置里打开 topK 分析：

```yaml
topk_metrics: [100, 500]
topk_output_dir: './topk_reports/'
topk_analysis_cols: ['raw_rating']
```

那么 DCNv2 会为每个 topK 导出一张 CSV，例如：

```text
top100.csv
top500.csv
```

二分类按预测分数排序。
多分类按各类别的 logits 排序。

如果配置了 `topk_analysis_cols`，还会额外输出对应分析列的 CSV，例如：

```text
top100_raw_rating.csv
top500_raw_rating.csv
```

要让这类分析列进入 dataloader、但又不进入模型，建议在 `dataset_config.yaml` 中把它们保留在 `feature_cols`，并设置为 `type: meta`。

### 6. 训练后会产出的文件

训练过程中和训练完成后，通常会看到这些输出：

1. 你指定数据目录下的 train/test CSV。
2. 预处理完成后 `data_root/dataset_id/` 下的 `feature_map.json`。
3. `model_root/<dataset_id>/` 下的模型 checkpoint。
4. 以 config 目录命名的实验结果汇总 CSV。
5. 如果开启了 topK 分析，还会有 `topk_output_dir` 下的 CSV 报表。

### 7. 最短复现命令序列

如果你只想最快跑通一套多分类 mock 数据训练流程，可以直接执行：

```bash
pip install -r requirements.txt
PYTHONPATH=. python model_zoo/DCNv2/generate_mock_nps_wide_data_multiclass.py --output_dir data/nps_wide_multiclass --num_samples 10000 --num_classes 3
PYTHONPATH=. python model_zoo/DCNv2/split_wide_csv.py --input_csv data/nps_wide_multiclass/wide.csv --train_csv data/nps_wide_multiclass/train.csv --test_csv data/nps_wide_multiclass/test.csv --test_ratio 0.3 --label_col label
PYTHONPATH=. python -m model_zoo.DCNv2.run_expid --config model_zoo/DCNv2/config --expid DCNv2_nps_multiclass --device cuda:0
```

如果使用真实数据，只是在这条链路前面多出“原始数据转换”和“dataset 配置生成”两步。
