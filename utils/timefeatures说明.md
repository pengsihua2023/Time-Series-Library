## timefeatures说明
以上代码主要是实现一组用于时间序列特征提取的工具类和函数。这些工具类通过从 `pandas.DatetimeIndex` 提取不同的时间特征，提供标准化的时间序列数据编码，方便后续模型使用这些时间特征进行预测或分析。

### 功能详细说明：

#### 1. **`TimeFeature` 基类**
- `TimeFeature` 是一个基类，定义了时间特征类的通用接口。
- 子类需要实现 `__call__` 方法，用于接收 `pd.DatetimeIndex` 对象并输出对应的时间特征（以 `numpy.ndarray` 格式）。

#### 2. **具体的时间特征类**
以下子类从时间索引中提取特定的时间信息，并将这些信息归一化到 `[-0.5, 0.5]` 范围内：

1. **`SecondOfMinute`**
   - 提取时间的秒数，归一化公式：`index.second / 59.0 - 0.5`。
   - 表示当前分钟中的秒，范围是 `[-0.5, 0.5]`。

2. **`MinuteOfHour`**
   - 提取时间的分钟数，归一化公式：`index.minute / 59.0 - 0.5`。
   - 表示当前小时中的分钟，范围是 `[-0.5, 0.5]`。

3. **`HourOfDay`**
   - 提取时间的小时数，归一化公式：`index.hour / 23.0 - 0.5`。
   - 表示当前天中的小时，范围是 `[-0.5, 0.5]`。

4. **`DayOfWeek`**
   - 提取时间的星期数，归一化公式：`index.dayofweek / 6.0 - 0.5`。
   - 表示当前日期是星期几，范围是 `[-0.5, 0.5]`。

5. **`DayOfMonth`**
   - 提取时间的月中的日期，归一化公式：`(index.day - 1) / 30.0 - 0.5`。
   - 表示当前日期是几号，范围是 `[-0.5, 0.5]`。

6. **`DayOfYear`**
   - 提取时间的年中的第几天，归一化公式：`(index.dayofyear - 1) / 365.0 - 0.5`。
   - 表示当前日期是这一年中的第几天，范围是 `[-0.5, 0.5]`。

7. **`MonthOfYear`**
   - 提取时间的月份，归一化公式：`(index.month - 1) / 11.0 - 0.5`。
   - 表示当前日期是第几个月，范围是 `[-0.5, 0.5]`。

8. **`WeekOfYear`**
   - 提取时间的年中的第几周，归一化公式：`(index.isocalendar().week - 1) / 52.0 - 0.5`。
   - 表示当前日期是这一年中的第几周，范围是 `[-0.5, 0.5]`。

#### 3. **`time_features_from_frequency_str` 函数**
- 根据输入的时间频率字符串（如 `"1D"`、`"1H"` 等），返回适合该频率的时间特征类的实例列表。
- 内部维护了一个 `features_by_offsets` 字典，将 `pandas.tseries.offsets` 对象映射到对应的时间特征类列表。
- 示例映射：
  - `offsets.YearEnd`：无特征
  - `offsets.MonthEnd`：`[MonthOfYear]`
  - `offsets.Day`：`[DayOfWeek, DayOfMonth, DayOfYear]`
  - `offsets.Hour`：`[HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]`
- 根据输入的时间频率字符串调用 `to_offset` 转化为偏移量类型，再返回对应的特征实例列表。

#### 4. **`time_features` 函数**
- 接受日期索引（`dates`，`pandas.DatetimeIndex`）和时间频率字符串（`freq`，默认为 `'h'` 表示小时）。
- 调用 `time_features_from_frequency_str(freq)` 获取特征类实例列表。
- 对每个实例调用其 `__call__` 方法，提取时间特征。
- 最终以二维数组的形式返回所有时间特征。

#### 5. **特性归一化的目的**
- 时间特征的归一化至 `[−0.5, 0.5]` 范围，有以下优点：
  - 减少数值范围差异对模型训练的影响。
  - 保持数值分布的均衡性，提高训练稳定性。

### 使用场景
1. **时间序列预测**：为时间序列数据生成特征输入，辅助模型学习时间周期性和趋势。
2. **时间特征分析**：对历史数据的时间维度进行分析，如分析季节性变化。
3. **深度学习模型的输入**：为神经网络提供结构化的时间特征。

### 示例
```python
import pandas as pd
from timefeatures import time_features

# 示例日期索引
dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="H")
features = time_features(dates, freq="H")

print(features.shape)  # 输出：(25, n_features)
```
