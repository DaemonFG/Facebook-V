- 数据来源：[Kaggle Recruiting Competition](https://www.kaggle.com/c/facebook-v-predicting-check-ins)

- 分析：
1 共两个数据集，train.csv和test.csv，包含row_id登记事件的id、x y坐标，accuracy定位准确性、time时间戳、place_id预测目标的id
2 根据数据特点，采用分类算法中的KNN，特征值：x y, accuracy, time，目标值place_id
3 由于数据量巨大，为了节省时间和算力，将x,y缩小，只保留(0<x<10 & 0<y<10)的数据，具备条件再使用完整数据。
4 时间戳不便于使用，将其转换成周日时分秒作为新的特征。
5 根据实际运算场景，将少于指定入住人数的位置删除，以提高预测准确性。
- 补充：
1 提升准确性的过程中删除了row_id
