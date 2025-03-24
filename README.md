# ML-MultiLoc_Dataset 数据集说明

## 概述

`ML-MultiLoc_Dataset` 是一个用于多标签分类任务的数据集，基于无线信号数据（RSSI 和 Phase）。数据集从多个 CSV 文件中提取数据，
按照 6:2:2 的比例划分为训练集、验证集和测试集，并保存为 PyTorch 张量文件（`.pt` 格式）。

保存的文件包括：
- 数据文件：`train_data.pt`、`val_data.pt`、`test_data.pt`
- 标签文件：`train_label.pt`、`val_label.pt`、`test_label.pt`

---

## 数据集结构

### 数据来源
- 数据来源于指定目录下的 `.csv` 文件，目录结构如下：
  ```
  dataset_dir/
  ├── one/
  │   ├── s_1.csv
  │   ├── s_2.csv
  │   └── ...
  ├── two/
  │   ├── s_1_2.csv
  │   └── ...
  └── three/
      ├── s_1_2_3.csv
      └── ...
  ```
- 每个 `.csv` 文件包含无线信号数据（RSSI 和 Phase），按天线和样本组织。

### 数据划分
- 数据集按以下比例划分：
  - **训练集**：60%
  - **验证集**：20%
  - **测试集**：20%

### 数据格式
- **数据文件**（`train_data.pt`、`val_data.pt`、`test_data.pt`）：
  - 形状：`(样本数, 13, 16, 2)`
  - 含义：
    - `样本数`：每个子集的样本数量（总数的 60%、20%、20%）。
    - `13`：13 个天线。
    - `16`： 16 个RFID Tag。
    - `2`： 2 个特征（RSSI 和 Phase）。
  - 数据类型：`torch.float32`
- **标签文件**（`train_label.pt`、`val_label.pt`、`test_label.pt`）：
  - 形状：`(样本数, 16)`
  - 含义：
    - `样本数`：与对应数据文件一致。
    - `16`：16 个位置的激活状态（0 或 1）。
    - 值为 `1` 表示该位置存在目标，值为 `0` 表示不存在目标。
  - 数据类型：`torch.float32`

---

## 文件列表

| 文件名          | 描述         | 形状               |
|-----------------|--------------|--------------------|
| `train_data.pt` | 训练集数据   | `(训练样本数, 13, 16, 2)` |
| `train_label.pt`| 训练集标签   | `(训练样本数, 16)`        |
| `val_data.pt`   | 验证集数据   | `(验证样本数, 13, 16, 2)` |
| `val_label.pt`  | 验证集标签   | `(验证样本数, 16)`        |
| `test_data.pt`  | 测试集数据   | `(测试样本数, 13, 16, 2)` |
| `test_label.pt` | 测试集标签   | `(测试样本数, 16)`        |

---

## 使用方法

### 依赖
- Python 3.x
- PyTorch
- NumPy

### 加载数据
使用 PyTorch 的 `torch.load` 方法加载数据集：
```python
import torch

# 加载训练集
train_data = torch.load('train_data.pt')
train_labels = torch.load('train_label.pt')

# 加载验证集
val_data = torch.load('val_data.pt')
val_labels = torch.load('val_label.pt')

# 加载测试集
test_data = torch.load('test_data.pt')
test_labels = torch.load('test_label.pt')

# 示例：打印训练集形状
print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
```

### 数据加载器
可以将数据与 PyTorch 的 `DataLoader` 结合使用：
```python
from torch.utils.data import TensorDataset, DataLoader

# 创建 TensorDataset
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 迭代数据
for batch_data, batch_labels in train_loader:
    print(batch_data.shape, batch_labels.shape)
    break
```

---

## 注意事项
1. **文件路径**：确保 `.pt` 文件位于当前工作目录，或在加载时指定正确路径。
2. **数据完整性**：加载前确认文件未被损坏，样本数与标签数应一致。
3. **内存需求**：大数据集可能需要较大内存，建议根据硬件条件调整批次大小。
