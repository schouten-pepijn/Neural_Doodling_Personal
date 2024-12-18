import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from matplotlib import pyplot as plt
import time

print(tf.__version__)
print(tf.config.list_physical_devices())

devices = ['/CPU:0', '/GPU:0']
epochs = 20
batch_size= 16

dataset = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = (
    dataset.load_data())

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_times = dict()
for device in devices:
    with tf.device(device):
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        
        inputs = keras.Input(shape=(28, 28))
        x = keras.layers.Flatten(input_shape=(28, 28))(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(10)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        print(model.summary())
        
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        start_time = time.time()
        
        model.fit(train_images, train_labels,
                  epochs=epochs,
                  batch_size=batch_size)
        
        end_time = time.time() - start_time
        
        train_times[device] = end_time
        
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    #%%
for key, value in train_times.items():
    print(f'Train time for {key} is {value:.4f}')
    
    

