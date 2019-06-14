import os
import tensorflow as tf
from convert import encode


def save_training_data(images, labels, filename):
    assert images.shape[0] == labels.shape[0]
    num_examples = images.shape[0]

    with tf.python_io.TFRecordWriter(filename) as writer:

        for index in range(num_examples):

            image = images[index]
            label = labels[index]
            example = encode(image, label)
            writer.write(example.SerializeToString())


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:len(x_train) // 2]
y_train = y_train[:len(y_train) // 2]

data_dir = os.path.expanduser("./data/")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

save_training_data(x_train, y_train, os.path.join(data_dir, "train_half.tfrecord"))
save_training_data(x_test, y_test, os.path.join(data_dir, "test.tfrecord"))
