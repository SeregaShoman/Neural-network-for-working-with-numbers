# Neural-network-for-working-with-numbers

Для работы использовал Python 3. 10 При поддержке библиотек:

-numpy
-matplotlib.pyplot
-sklearn
-tensorflow

Запускать через командную строку python number.py и python print_number.py

ЧИСЛОВЫЕ ЗНАЧЕНИЯ number.py разработана на чистом питоне и numpy, работает от значения x которое передается в строке 160 принимает значение класса из строки 164. Обучалась с помощью датасета sklearn iris (Только там я смог найти нормальные значения для её обучения)

ГРАФИЧЕСКИЕ ЗНАЧЕНИЯ print_number.py разработана на tensorflow и keras, она обучена на датасете sklearn digits (60 000 наборов символов представленных в формате значения от 0 до 255 пикселей серого оттенка). Она обучается и распознаёт все эти символы т.е если ей передать значение какого то определённого например 1, то она его распознает с вероятностью > 97 процентов.
