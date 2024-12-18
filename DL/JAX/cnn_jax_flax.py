import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import jax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax import linen as nn
from flax.training import train_state
import dm_pix as pix
from flax.training import checkpoints

print(jax.local_devices())
print(tf.__version__)

BASE_DIR = os.path.join("CAT_DOGS_IMG", "dog vs cat",
                        "dataset", "training_set")
TEST_DIR = os.path.join("CAT_DOGS_IMG", "dog vs cat",
                        "dataset", "test_set")
BATCH_SIZE = 64
IMG_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE
LR = 1e-5
NUM_EPOCHS = 50

#%% DATA
training_set = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR, validation_split=0.2, batch_size=BATCH_SIZE,
    subset="training", seed=87)
validation_set = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR, validation_split=0.2, batch_size=BATCH_SIZE,
    subset="validation", seed=87)
test_set = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, batch_size=BATCH_SIZE)

# scaling
resize_rescale = tf.keras.Sequential(
    [
         tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
         tf.keras.layers.Rescaling(1.0 / 255),
     ]
    )

image_example = next(iter(training_set))[0][0][None]
print("image shape: ", image_example.shape)

scaled_example = resize_rescale(image_example)
print("scaled image shape: ", scaled_example.shape)

training_set_temp = training_set.map(lambda img, y: (resize_rescale(img), y))
validation_set_temp = training_set.map(lambda img, y: (resize_rescale(img), y))
test_set_temp = training_set.map(lambda img, y: (resize_rescale(img), y))

fig, axs = plt.subplots(3, 3, figsize=(10,10))
for images, _ in training_set_temp.take(1):
    for i, ax in zip(range(9), axs.flat):
        ax.imshow(np.array(images[i]))
        ax.axis('off')
plt.show()

# augmentation
rng = jax.random.PRNGKey(87)
rng, inp_rng, init_rng = jax.random.split(rng, 3)

@jax.jit
def data_augmentation(img):
    new_img = pix.random_brightness(image=img, max_delta=0.2,
                                      key=inp_rng)
    new_img = pix.flip_up_down(image=new_img)
    new_img = pix.flip_left_right(image=new_img)
    new_img = pix.rot90(k=1, image=new_img)
    return new_img

plt.figure(figsize=(10,10))
augmented_images = []
for images, _ in training_set_temp.take(1):
    for i in range(9):
        augmented_image = data_augmentation(np.array(images[i],
                                                     dtype=jnp.float32))
        augmented_images.append(augmented_image)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(augmented_images[i])
        plt.axis('off')
plt.show()        

# prepare dataset
jit_data_augmentation = jax.vmap(data_augmentation)
def prepare(ds, shuffle=False):
    ds = ds.map(lambda x, y: (resize_rescale(x), y),
                num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(training_set, shuffle=True)
val_ds = prepare(validation_set)
test_ds = prepare(test_set)

def get_batches(ds):
    data = ds.prefetch(1)
    return tfds.as_numpy(data)

training_data = get_batches(train_ds)
validation_data = get_batches(val_ds)
test_data = get_batches(test_ds)

# create cnn jax with flax
class_names = training_set.class_names
num_classes = len(class_names)

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=num_classes)(x)
        return x
    
# initialize model
model = CNN()
inp = jnp.ones([1, IMG_SIZE, IMG_SIZE, 3])
params = model.init(init_rng, inp)
print(params)

print(model.apply(params, inp))

# training state
optimizer = optax.adam(learning_rate=LR)
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer)

# compute metrics
def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    data_input = jit_data_augmentation(data_input)
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input)
    # Calculate the loss and accuracy
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    #uncomment the line below for multiclass classification
    # loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    loss = optax.sigmoid_binary_cross_entropy(logits, labels_onehot).mean()
    # comment the line above for multiclass problems
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, acc

batch = next(iter(training_data))
print(calculate_loss_acc(model_state, model_state.params, batch))

# create training step
@jax.jit
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss_acc,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=True,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc

# create evaluation step
@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    loss, acc = calculate_loss_acc(state, state.params, batch)
    return loss, acc


# training loop
training_accuracy = []
training_loss = []

testing_loss = []
testing_accuracy = []


def train_model(state, train_loader, test_loader, num_epochs=30):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_batch_loss, train_batch_accuracy = [], []
        val_batch_loss, val_batch_accuracy = [], []

        for train_batch in train_loader:
            state, loss, acc = train_step(state, train_batch)
            train_batch_loss.append(loss)
            train_batch_accuracy.append(acc)

        for val_batch in test_loader:
            val_loss, val_acc = eval_step(state, val_batch)

            val_batch_loss.append(val_loss)
            val_batch_accuracy.append(val_acc)

        # Loss for the current epoch
        epoch_train_loss = np.mean(train_batch_loss)
        epoch_val_loss = np.mean(val_batch_loss)

        # Accuracy for the current epoch
        epoch_train_acc = np.mean(train_batch_accuracy)
        epoch_val_acc = np.mean(val_batch_accuracy)

        testing_loss.append(epoch_val_loss)
        testing_accuracy.append(epoch_val_acc)

        training_loss.append(epoch_train_loss)
        training_accuracy.append(epoch_train_acc)

        print(
            f"Epoch: {epoch + 1}, loss: {epoch_train_loss:.2f}, acc: {epoch_train_acc:.2f} val loss: {epoch_val_loss:.2f} val acc {epoch_val_acc:.2f} "
        )

    return state


trained_model_state = train_model(
    model_state, training_data, validation_data, num_epochs=NUM_EPOCHS)

# evaluation
metrics_df = pd.DataFrame(np.array(training_accuracy), columns=["accuracy"])
metrics_df["val_accuracy"] = np.array(testing_accuracy)
metrics_df["loss"] = np.array(training_loss)
metrics_df["val_loss"] = np.array(testing_loss)
metrics_df[["loss", "val_loss"]].plot()
metrics_df[["accuracy", "val_accuracy"]].plot()


# storing and loading
checkpoints.save_checkpoint(
    ckpt_dir="/content/my_checkpoints/",  # Folder to save checkpoint in
    target=trained_model_state,  # What to save. To only save parameters, use model_state.params
    step=100,  # Training step or other metric to save best model on
    prefix="my_model",  # Checkpoint file name prefix
    overwrite=True,  # Overwrite existing checkpoint files
)
