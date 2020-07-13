# Import TensorFlow and TensorFlow Datasets

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os


# In[3]:


print(tf.__version__)


# ## Download the dataset

# Download the MNIST dataset and load it from [TensorFlow Datasets](https://www.tensorflow.org/datasets). This returns a dataset in `tf.data` format.

# Setting `with_info` to `True` includes the metadata for the entire dataset, which is being saved here to `info`.
# Among other things, this metadata object includes the number of train and test examples. 
# 

# In[4]:


datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']


# ## Define distribution strategy

# Create a `MirroredStrategy` object. This will handle distribution, and provides a context manager (`tf.distribute.MirroredStrategy.scope`) to build your model inside.

# In[5]:


strategy = tf.distribute.MirroredStrategy()


# In[6]:


print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# ## Setup input pipeline

# When training a model with multiple GPUs, you can use the extra computing power effectively by increasing the batch size. In general, use the largest batch size that fits the GPU memory, and tune the learning rate accordingly.

# In[7]:


# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# Pixel values, which are 0-255, [have to be normalized to the 0-1 range](https://en.wikipedia.org/wiki/Feature_scaling). Define this scale in a function.

# In[8]:


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


# Apply this function to the training and test data, shuffle the training data, and [batch it for training](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch). Notice we are also keeping an in-memory cache of the training data to improve performance.
# 

# In[9]:


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)


# ## Create the model

# Create and compile the Keras model in the context of `strategy.scope`.

# In[10]:


with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])


# ## Define the callbacks
# 

# The callbacks used here are:
# 
# *   *TensorBoard*: This callback writes a log for TensorBoard which allows you to visualize the graphs.
# *   *Model Checkpoint*: This callback saves the model after every epoch.
# *   *Learning Rate Scheduler*: Using this callback, you can schedule the learning rate to change after every epoch/batch.
# 
# For illustrative purposes, add a print callback to display the *learning rate* in the notebook.

# In[11]:


# Define the checkpoint directory to store the checkpoints

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# In[12]:


# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5


# In[13]:


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))


# In[14]:


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]


# ## Train and evaluate

# Now, train the model in the usual way, calling `fit` on the model and passing in the dataset created at the beginning of the tutorial. This step is the same whether you are distributing the training or not.
# 

# In[15]:



print("WUICHAK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
model.fit(train_dataset, epochs=12, callbacks=callbacks)

