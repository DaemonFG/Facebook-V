"""
k近邻算法

定义：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，
    则该样本也属于这个类别。

欧式距离公式：a(a1,a2,a3), b(b1,b2,b3)
            √([(𝑎1−𝑏1)]^2+[(𝑎2−𝑏2)]^2+[(𝑎3−𝑏3)]^2 )

需要标准化处理

案例：预测入住位置
数据来源：https://www.kaggle.com/c/facebook-v-predicting-check-ins

案例分析：
1、共两个数据集，train.csv和test.csv，包含row_id登记事件的id、x y坐标，accuracy定位准确性、time时间戳、place_id预测目标的id
2、根据数据特点，采用分类算法中的KNN，特征值：x y, accuracy, time，目标值place_id
3、由于数据量巨大，为了节省时间和算力，将x,y缩小，只保留(0<x<10 & 0<y<10)的数据，具备条件再使用完整数据。
4、时间戳不便于使用，将其转换成周日时分秒作为新的特征。
5、根据实际运算场景，将少于指定入住人数的位置删除，以提高预测准确性。
补充：
1、提升准确性的过程中删除了row_id
2、进行网格搜索调优
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd


def knncls():
    """
    预测入住位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv("./train.csv")
    # print(data.head(10))

    # 处理数据
    # 1、缩小数据，查询数据筛选
    data = data.query("x>0 & x<10 & y>0 & y<10")

    # 2、处理时间数据
    time_value = pd.to_datetime(data['time'], unit='s')
    # print(time_value)

    # 3、把日期格式转换为字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 4、构造一些特征
    data.loc[:, 'day'] = time_value.day
    data.loc[:, 'hour'] = time_value.hour
    data.loc[:, 'weekday'] = time_value.weekday

    # 5、删除时间戳
    data.drop(['time'], axis=1)  # pandas中axis=1表示列，sklearn中表示行
    # print(data)

    # 6、把入住数量少于3个目标位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index  # 将索引变成place_id，原索引位置重置
    data = data[data['place_id']].isin(tf.place_id)  # 筛选data

    # 7、删除登记事件的id
    data.drop(['raw_id'], axis=1)

    # 8、去除数据当中的特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    # 进行数据分割，训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程(标准化)
    std = StandardScaler()
    # 对测试集和预测集特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    # fit输入数据，predict预测目标值，score得到准确性
    # knn.fit(x_train, y_train)
    # 得出预测结果
    # y_predict = knn.predict(x_test)
    # print("预测目标入住位置为：", y_predict)
    # 得出准确率
    # print("预测准确率：", knn.score(x_test, y_test))

    """
    1、交叉验证
    为了让被评估模型更加准确可信，将拿到的数据，分为训练和验证集。例如，将数据分成5份，其中一份作为验证集。
    然后经过5次(组)的测试，每次都更换不同的验证集。即得到5组模型的结果，取平均值作为最终结果。又称5折交叉验证。
    一般来说10折最佳。

    2、超参数搜索-网格搜索
    通常情况下，有很多参数是需要手动指定的(如k-近邻算法中的K值)，这种叫超参数。
    但是手动过程繁杂，所以需要对模型预设几种超参数组合。每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

    sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
    对估计器的指定参数值进行详尽搜索

    estimator：估计器对象
    param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
    cv：指定几折交叉验证
    fit：输入训练数据
    score：准确率
    结果分析：
    best_score_:在交叉验证中测试的最好结果
    best_estimator_：最好的参数模型
    cv_results_:每次交叉验证后的测试集准确率结果和训练集准确率结果
    """
    # 构造一些参数值进行搜索
    param = {"n_neighbors": [3, 7, 10]}

    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=10)
    gc.fit(x_train, y_train)

    # 预测准确率
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证中的最好结果：", gc.best_score_)
    print("选择最好的模型：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)
    return None


if __name__ == '__main__':
    knncls()
