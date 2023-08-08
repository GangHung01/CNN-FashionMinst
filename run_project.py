from DeepLearningFashion.predict import model_predict
from DeepLearningFashion.train import run_train

if __name__ == '__main__':
    #图片经过预处理后，展示训练集前100张图片，经过模型搭建，以及训练，并保存模型。并展示训练中的准确率和损失。
    run_train()
    #加载模型，并经行预测。并展示测试集前一百张图和预测前十张图片以及结果。
    model_predict()