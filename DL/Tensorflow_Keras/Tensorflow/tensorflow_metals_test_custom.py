import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

print (f'TF version: {tf.__version__}')
print(tf.config.list_physical_devices())

device = '/CPU:0'
dtype = 'float32'
batch_size = 32
epochs = 10

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(dtype)
x_test = x_test[..., tf.newaxis].astype(dtype)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(batch_size)


class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        outputs = self.d2(x)
        return outputs
    
model = MyModel()

criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

train_loss = keras.metrics.Mean(name='train_loss')
train_acc = keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

test_loss = keras.metrics.Mean(name='test_loss')
test_acc = keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')

@tf.function(jit_compile=True)
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = criterion(labels, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
     
    train_loss.update_state(loss)
    train_acc.update_state(labels, preds)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    loss = criterion(labels, predictions)

    test_loss.update_state(loss)
    test_accuracy.update_state(labels, predictions)
    
with tf.device(device):
    for epoch in range(epochs):
        train_loss.reset_state()
        train_acc.reset_state()
        test_loss.reset_state()
        test_acc.reset_state()
        
        for images, labels in train_ds:
            train_step(images, labels)
        
        for images, labels in test_ds:
            test_step(images, labels)
            
        print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():0.2f}, '
        f'Accuracy: {train_acc.result() * 100:0.2f}, '
        f'Test Loss: {test_loss.result():0.2f}, '
        f'Test Accuracy: {test_acc.result() * 100:0.2f}'
      )   