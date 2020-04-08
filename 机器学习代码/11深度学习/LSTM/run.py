#!/usr/bin/env python
#-*- coding: utf-8 -*-
import lstm
import time
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
#from keras.models import load_model

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

if __name__=='__main__':
    global_start_time = time.time()
    seq_len = 100

    X_train, y_train, X_test, y_test = lstm.load_data('small_data.csv', seq_len, True)
#    训练模型
    filepath = "model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model = lstm.build_model([1, 100, 200, 1])
    model.fit(X_train, y_train,batch_size=512,nb_epoch=1,validation_split=0.05, callbacks=callbacks_list)
    print(model.summary())
#    加载模型
#    model = load_model("model.h5")
    predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 100)

    print('duration (s) : ', time.time() - global_start_time)
    plot_results_multiple(predictions, y_test, 100)

