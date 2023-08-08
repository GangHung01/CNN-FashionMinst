import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# 定义方法用于读取数据
def get_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


# 展示前100张图片（25X25）
def show_image(x_train):
    plt.figure()
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()


# 定义方法:构建模型,采用Alextnet网络结构
def build_model():
    # 初始化模型对象
    model = models.Sequential()
    # 卷积层1
    model.add(layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(5, 5), filters=64, strides=(1, 1), padding='same',
                            activation='relu'))
    # 优化层1
    model.add(layers.BatchNormalization(axis=-1, epsilon=0.001, momentum=0.99, center=True, trainable=True))
    # 池化层1
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # 卷积层2
    model.add(layers.Conv2D(kernel_size=(5, 5), filters=64, strides=(1, 1), padding='same', activation='relu'))
    # 优化层2
    model.add(layers.BatchNormalization(axis=-1, epsilon=0.001, momentum=0.99, center=True, trainable=True))
    # 池化层2
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # 卷积层3
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=128, strides=(1, 1), padding='same', activation='relu'))
    # 卷积层4
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=128, strides=(1, 1), padding='same', activation='relu'))
    # 卷积层5
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=128, strides=(1, 1), padding='same', activation='relu'))
    # 池化层3
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # 拉直层
    model.add(layers.Flatten())
    # 全连接层1
    model.add(layers.Dense(2048, activation='relu'))
    # 全连接层2
    model.add(layers.Dense(2048, activation='relu'))
    # softmax层
    model.add(layers.Dense(activation='softmax', units=10))
    model.summary()
    return model


def run_train():
    (x_train, y_train), (x_test, y_test) = get_data()
    show_image(x_train)
    model = build_model()
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=256)
    # 对训练过程的准确率和损失可视化
    fig = plt.figure(figsize=(20, 10))
    plt.title('训练')
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['font.size'] = 12
    plt.plot(history.history['accuracy'], color='red')
    plt.plot(history.history['loss'], color='green')
    plt.legend(['准确率', '损失'])
    plt.show()
    # 保存模型
    model.save('./save/model.h5')
