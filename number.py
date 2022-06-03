import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
data = datasets.load_iris()
datasets = [ ( data.data[ i ][ None, ... ], data.target[ i ] ) for i in range( len( data.target ) ) ]
#print(datasets) ЭТО ПОКАЖЕТ ВАМ МАССИВ ИЗ МАТРИЦЫ 13 СТОЛБИКОВ 5 СТРОК 
#exit()

INPUT_DIM = 4 # ВХОДНЫЕ ЗНАЧЕНИЯ
OUT_DIM = 3#Выходные значения
H_DIM = 10 #Нейроны

'''
ДВЕ МАТРИЦЫ ВЕСОВ И ДВА ВЕКТОРА СМЕЩЕНИЯ
ПОТОМУ ЧТО У НАС 2 СЛОЯ
'''
W1 = np.random.rand( INPUT_DIM, H_DIM )
b1 = np.random.rand( 1, H_DIM )
W2 = np.random.rand( H_DIM, OUT_DIM )
b2 = np.random.rand( 1, OUT_DIM )

#Равномерное распределение с диапазоном
W1 = ( W1 - 0.5 ) * 2 * np.sqrt( 1/INPUT_DIM )
b1 = ( b1 - 0.5 ) * 2 * np.sqrt( 1/INPUT_DIM )
W2 = ( W2 - 0.5 ) * 2 * np.sqrt( 1/H_DIM )
b2 = ( b2 - 0.5 ) * 2 * np.sqrt( 1/H_DIM )

ALPHA = 0.0002#Скорость обучения
EPOCH =  500#Кол во итераций по датасету
BATCH_SIZE = 50#Группа образцов для градиента
loss_arr = []

def predict( x ):
    t1 = x @ W1 + b1
    h1 = relu( t1 )   
    #Второй слой 
    t2 = h1 @ W2 + b2
    z = softmax( t2 )
    return z


def relu( t ):#Активация
    return np.maximum( t, 0 )#Максимум из пришедшего значения и нуля

def softmax( t ):
    out = np.exp( t )
    return out / np.sum( out )

def softmax_batch( t ):
    #Вычисляем экспоненту
    out = np.exp( t )
    #Суммируем лишь по 1 измерению
    return out / np.sum( out, axis=1, keepdims=True )

def sparse_cross_entropy( z, y ):
    return -np.log( z[ 0, y ] )

def to_full( y, num_classes ):
    y_full = np.zeros( ( 1, num_classes ) )#Вектор строка из нулей
    y_full[ 0, y ] = 1#присвоили единичку
    return y_full

def to_full_batch( y, num_classes ):
    """
    Создаем матрицу из нулей 
    и расставляем в ней единички в нужных позициях
    Батч из полных y
    """
    y_full = np.zeros( ( len( y ), num_classes ) )
    for j, yj in enumerate( y ):
        y_full[ j, yj ] = 1
    return y_full

def sparse_cross_entropy_batch( z, y ):
    """
    Считаем кроссэнтропию для каждого элемента батча
    Для каждого вектора строки из z и правильного индекса y
    """
    return -np.log( np.array( [ z [ j, y [ j ] ] for j in range( len( y ) ) ] ) )


def relu_deriv( t ):
    """
    t вектор
    Сравниваем т с нулём получаем True или False 
    и приравнивает к float 
    т.е получаем 0 или 1
    """
    return ( t >= 0 ).astype( float )

for ep in range( EPOCH ):#Цикл для эпох
    #Перемешиваем на каждой эпохе чтоб примеры показывались в новых порядках
    random.shuffle( datasets )

    """
    Тут мы выводим все правильные игрики которые сеть учла
    """
    #Вычисляем точность т.е правильно угаданные y
    def calc_accuracy():
        correct = 0
        for x, y in datasets:
            z = predict( x )
            y_pred = np.argmax( z )#Выбираем индекс класса
            if y_pred == y:#Проверяем совпало ли
                correct += 1
        acc = correct/len(datasets)
        return acc
    
    accuracy = calc_accuracy()
    print("Приближенное значение за 1 эпоху итераций: ", accuracy)
    for i in range( len( datasets ) // BATCH_SIZE ):#Цикл по всем элементам датасета
        

        #подвыборки датасета
        batch_x, batch_y = zip( *datasets[ i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE ] )
        x = np.concatenate( batch_x, axis=0 )
        y = np.array( batch_y )
        #По индексу получаем значение(x) и приближенное к нему значение

        # Распространение
        #Первый слой
        t1 = x @ W1 + b1
        h1 = relu( t1 )   
        #Второй слой
        t2 = h1 @ W2 + b2
        z = softmax_batch( t2 )
        E = np.sum( sparse_cross_entropy_batch( z, y ) )#Ошибка от вероятности и правильного ответа
    
        # Обратное расспространение ГРАДИЕНТ ФОРМУЛА
        y_full = to_full_batch( y, OUT_DIM )#ФУНКЦИЯ to_full превращает индекс правильного класса в нужное расспределение
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum( dE_dt2, axis=0, keepdims=True )#Суммирование по батч измерению
        dE_dh1 = dE_dt2 @ W2.T 
        dE_dt1 = dE_dh1 * relu_deriv( t1 )#Производная
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum( dE_dt1, axis=0, keepdims=True )

        """ 
        Обновление
        Примерно так и выглядит одна итерация алгоритма обучения
        """
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        """
        Анализ обучения(Зависимость ошибки от итераций)
        Помешаем скаляр ошибки в список и ресуем его график после обучения
        """
        loss_arr.append( E )

#Если играть с этими параметрами то график выборки обучения может быть интереснее
plt.plot( loss_arr )
plt.show()

x = np.array( [ 1, 1, 4, 4 ] )

probs = predict( x )
pred_class = np.argmax( probs )
class_names = [ '1', '2', '3', '4' ]
print(" ")
print( 'Выбранный нейросетью класс', class_names[ pred_class ] )