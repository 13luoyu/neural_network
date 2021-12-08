# import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

#采用神经网络训练并测试MNIST数据集，输入一个数字图片，识别出图片中的数字。

# - 训练集http://www.pjreddie.com/media/files/mnist_train.csv
# - 测试集 http://www.pjreddie.com/media/files/mnist_test.csv
#
# - MNIST测试数据集中的10条记录——https://raw.githubusercontent.com/
# makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/
# mnist_test_10.csv
# - MNIST训练数据集中的100条记录——https://raw.githubusercontent.com/
# makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/
# mnist_train_100.csv

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 创建层，指定学习率
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 指定边权重
        # 法1，随机-0.5~0.5的值
        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
        # 法2，正态分布，均值=0，方差=1/根号(传入链接数目)
        self.wih = np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes))

        # sigmoid激活函数，使用lambda表达式
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # 针对给定的训练样本计算输出
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差 = 目标 - 实际输出
        output_errors = targets - final_outputs
        # 隐藏层误差 = weights hidden_output的转置 * 输出层误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # 依据误差更新权重
        # w的变化 =α * dE/dw(j,k) =
        # α * Ek * sigmoid(Σw(j,k)*Oj) * (1-sigmoid(Σw(j,k)*Oj)) · Oj的转置
        # α是学习率，Ek是输出层误差output_errors，sigmoid(Σw(j,k)*Oj)是输出层输出final_outputs，
        # Oj是隐藏层输出hidden_outputs
        # *是正常的对应元素的乘法，·是矩阵乘法
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs)
        )
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs)
        )

        pass

    def query(self, inputs_list):
        # 转换输入数组为二维数组
        inputs = np.array(inputs_list, ndmin=2).T  # T转置
        # 计算输入隐藏层的信号 Xhidden = Winput_hidden * I
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算激活后的隐藏层输出 Ohidden = sigmoid(Xhidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输入输出层的信号
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算激活后的输出层输出
        final_outpus = self.activation_function(final_inputs)
        return final_outpus

        pass


input_nodes = 784  # 28*28
hidden_nodes = 200
output_nodes = 10  # 0-9
learning_rate = 0.1

n=neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# data_file = open("csv/mnist_train_100.csv", 'r')
data_file = open("csv/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# 反复训练次数
epochs = 1

for e in range(epochs):
    for record in data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# data_file = open("csv/mnist_test_10.csv", 'r')
data_file = open("csv/mnist_test.csv", 'r')
data_list = data_file.readlines()
data_file.close()

scorecard = []
for record in data_list:
    all_values = record.split(',')
    # print(all_values[0])
    # image_array = np.asfarray(all_values[1:]).reshape((28,28))
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # plt.show()
    #
    # print(n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))

    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

scorecard_array=np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)










