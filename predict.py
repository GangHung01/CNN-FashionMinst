import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from DeepLearningFashion.train import get_data, show_image

(x_train, y_train), (x_test, y_test) = get_data()
# 输出值对应的标签
label_dict = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
              8: "Bag", 9: "Ankle boot"}


# 方便打印预测值的前十个值
def add_list(predict_results):
    predict = []
    for i in range(10):
        a = predict_results[i]
        predict.append(a)
    return predict


# 预测结果可视化：定义显示图像数据及其对应标签的函数 （使用cifra10实验中老师的代码模板）
def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  predictions,  # 预测值列表
                                  index,  # 从第index个开始显示
                                  num=5):  # 缺省一次显示5幅
    fig = plt.gcf()  # 获取当前图表，Get Current Figure
    fig.set_size_inches(12, 6)  # 1英寸等于 2.54 cm
    if num > 10:
        num = 10  # 最多显示10个子图
    for i in range(0, num):
        ax = plt.subplot(2, 5, i + 1)  # 获取当前要处理的子图

        ax.imshow(images[index],  # 显示第index个图像
                  cmap='binary')

        title = str(i) + ',' + label_dict[np.argmax(labels[index])]  # 构建该图上要显示的title信息
        if len(predictions) > 0:
            title += ' => ' + label_dict[np.argmax(predictions[index])]

        ax.set_title(title, fontsize=10)  # 显示图上的title信息
        index += 1
    plt.show()


def model_predict():
    # 1,加载训练好的模型
    model = tf.keras.models.load_model('./save/model.h5')
    # 2,执行预测
    predict_results = model.predict(x=x_test, batch_size=256)
    # 3,展示预测的前十个值，以数字作为分类标准
    predict = np.argmax(predict_results, axis=-1)
    predict = add_list(predict)
    print(predict)
    # 4,展示预测集前100张图片（25X25）
    show_image(x_test)
    # 5,展示预测的前10张图片的结果（因显示大小有限，因此选择前十张）
    plot_images_labels_prediction(x_test, y_test, predict_results, 0, 10)


if __name__ == '__main__':
    # 加载模型，并经行预测。并展示测试集前一百张图和预测前十张图片以及结果。
    model_predict()
