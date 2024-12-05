## metrics说明
上述代码定义了一个名为 **`metrics.py`** 的模块，其中包含多个用于评估预测结果与真实值之间误差的评价指标函数。这些指标广泛用于回归任务中，帮助研究人员和工程师评估模型的预测性能。

以下是代码中各函数的详细功能说明：

---

### **1. `RSE(pred, true)`**
- **全称**: Root Relative Squared Error（相对平方误差的平方根）
- **功能**: 计算预测值与真实值之间的相对平方误差，归一化误差以便比较不同数据集上的模型表现。
- **公式**:  
![image](https://github.com/user-attachments/assets/d981420c-1774-4589-9c5b-4ebd4708087c)  

- **解读**: 
  - 分子表示预测值与真实值的总平方误差。
  - 分母是基于真实值均值的平方误差，起到归一化的作用。
  - 值越小，模型预测的效果越好。

---

### **2. `CORR(pred, true)`**
- **全称**: Correlation Coefficient（相关系数）
- **功能**: 计算预测值与真实值之间的相关性，用于评估两者变化趋势的相似度。
- **公式**:  
![image](https://github.com/user-attachments/assets/3decc0a2-3795-4d09-b987-e624f0365cd2)  

- **解读**: 
  - 返回值介于 -1 和 1 之间：
    - 值接近 1 表示强正相关。
    - 值接近 -1 表示强负相关。
    - 值接近 0 表示无相关性。
  - 对高相关模型的预测非常重要。

---

### **3. `MAE(pred, true)`**
- **全称**: Mean Absolute Error（平均绝对误差）
- **功能**: 测量预测值与真实值之间绝对误差的平均值。
- **公式**:  
![image](https://github.com/user-attachments/assets/c272c744-9c8d-4d80-be76-bb9f5119a157)  

- **解读**: 
  - 反映了预测值与真实值的平均偏差。
  - 值越小，模型越精确。

---

### **4. `MSE(pred, true)`**
- **全称**: Mean Squared Error（均方误差）
- **功能**: 计算预测值与真实值之间误差的平方平均值。
- **公式**:  
![image](https://github.com/user-attachments/assets/2a33df0e-c779-45f9-abff-4d66117b2714)  

- **解读**: 
  - 强调较大的误差，因为平方放大了差值。
  - 常用于衡量模型的总体预测性能。

---

### **5. `RMSE(pred, true)`**
- **全称**: Root Mean Squared Error（均方根误差）
- **功能**: 计算 MSE 的平方根，将误差单位与原始数据保持一致。
- **公式**:  
![image](https://github.com/user-attachments/assets/bf1afacd-c410-4dbb-8b58-228b591c1cdf)  

- **解读**: 
  - 与 MSE 类似，但由于对误差平方根的操作，其值更加直观。
  - 值越小，模型性能越好。

---

### **6. `MAPE(pred, true)`**
- **全称**: Mean Absolute Percentage Error（平均绝对百分比误差）
- **功能**: 测量预测值相对于真实值的平均绝对百分比误差。
- **公式**:  
![image](https://github.com/user-attachments/assets/116d818d-97fd-4420-9699-b9f873be191d)  

- **解读**: 
  - 提供了一个无量纲的误差指标。
  - 通常用于数据真实值范围变化较大时。

---

### **7. `MSPE(pred, true)`**
- **全称**: Mean Squared Percentage Error（均方百分比误差）
- **功能**: 测量预测值相对于真实值的平方百分比误差的平均值。
- **公式**:  
![image](https://github.com/user-attachments/assets/8bfe2159-07a6-482b-968a-f3703c8081d1)   

- **解读**: 
  - 类似于 MAPE，但对较大的误差更加敏感。

---

### **8. `metric(pred, true)`**
- **功能**: 综合计算多个指标，并将结果返回为一个元组。
- **实现**: 
  ```python
  mae = MAE(pred, true)
  mse = MSE(pred, true)
  rmse = RMSE(pred, true)
  mape = MAPE(pred, true)
  mspe = MSPE(pred, true)

  return mae, mse, rmse, mape, mspe
  ```
- **用途**: 一次性返回多个评价指标的值，便于分析和比较模型表现。

---

### **总结**
这段代码实现了多种评价指标，覆盖了从误差量化（如 MAE, MSE）到趋势相关性分析（如 CORR）的多个方面。这些函数可以独立使用，也可以通过 `metric()` 函数一次性输出所有结果，适用于回归任务或其他需要评估预测性能的场景。
