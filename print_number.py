import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()#Две выборки мейновая и тестовая

#Нормируем значения (от 0 до 1)
x_train = x_train / 255
x_test = x_test / 255

#Вытягиваем изображения
x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])#Набор обучающих данных
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

"""
Преобразуем правильный ответ в вектор 
Например
x = 5
=>
y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ]
"""
y_train = to_categorical(y_train, 10)

#МОДЕЛЬ ПОЛНОСВЯЗННОГО СЛОЯ НЕЙРОНКИ

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate#relu
        self.fl_init = False
 
    def __call__(self, x):
        if not self.fl_init:
            #Начальные весовые категорий
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
 
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
 
            self.fl_init = True
 
        y = x @ self.w + self.b#Входные суммы на каждом нейроне
 
        if self.activate == "relu":
            return tf.nn.relu(y)#Активируем входные суммы
        elif self.activate == "softmax":
            return tf.nn.softmax(y)#Активируем входные суммы
 
        return y
#2 СЛОЯ
layer_1 = DenseNN(128)#Входной на 128 нейронов
layer_2 = DenseNN(10, activate="softmax")#Выходной на 10 и активацию через софтмакс

def predict(x):#Пропускаем вектор х через нейронную сеть
    y = layer_1(x)
    y = layer_2(y)
    return y

"""
ОБУЧЕНИЕ НЭЙРОННОЙ СЕТИ
Через градиентный спуск
и Кроссэнтропию
"""

#Применяем кроссэнтропию сразу к набору данных а не к отдельным наблюдениям и усредняем
cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
#оптимизатор для градиентного спуска
opt = tf.optimizers.Adam(learning_rate=0.001)

#Вспомогательные параметры для обучения
BATCH_SIZE = 32#Батч
EPOCH = 10#Эпохи
TOTAL = x_train.shape[0]# Размер обучающей выборки

#Разбиваем на батчи
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#Перемешиваем и группируем по батчам
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

#Цикл обучения

for n in range(EPOCH):#Эпохи перебераем
    loss = 0#Сохраняем суммарные значения
    for x_batch, y_batch in train_dataset:#Перебираем батчи
        
        with tf.GradientTape() as tape:
            f_loss = cross_entropy(y_batch, predict(x_batch))
 
        loss += f_loss
        grads = tape.gradient(f_loss, [layer_1.trainable_variables, layer_2.trainable_variables])
        #Применяем градиенты к обучающим параметрам
        opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
        opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))
    print("Параметор хода обучения: ",loss.numpy())#Суммарное значение потери

#Определяем качество работы
y = predict(x_test)
y2 = tf.argmax(y, axis=1).numpy()#Преобразовываем вектора в обычные числа
acc = len(y_test[y_test == y2])/y_test.shape[0] * 100#Сравнение что получилось и что должно быть
print(acc,"% классификаций изображенй верно расспознанны!")