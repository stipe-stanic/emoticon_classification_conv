import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib
import numpy.typing as npt
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,\
    ConfusionMatrixDisplay, mean_absolute_error

from visualization import show_images_from_set, show_augmented_images_from_set, plot_model_metrics

# INFO and WARNING messages are not printed
tf.get_logger().setLevel('ERROR')

# from version >= 2.11.0 TensorFlow dropped native support for Windows
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # 0

data_dir = pathlib.Path('train_data')

# count of all images present in training set
image_count = len(list(data_dir.glob("*/*.png")))
print(image_count)

angry_faces = list(data_dir.glob('angry_face/*'))
# PIL.Image.open(str(angry_faces[0])).show()

# loading
batch_size = 32
img_height = 120
img_width = 120

# shape(None, 120, 120, 3) dtype=tf.float32
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
print(class_names)

# visualization
show_images_from_set(train_ds, class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)    # (32, 120, 120, 3)
    print(labels_batch.shape)   # (32,)
    break

AUTOTUNE = tf.data.AUTOTUNE  # tunes values dynamically at runtime

# optimized
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# train_data augmentation layer, included in model
data_augmentation = keras.Sequential([
    layers.RandomFlip(
        "horizontal",
        input_shape=(img_height, img_width, 3)
    ),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

show_augmented_images_from_set(train_ds, data_augmentation)

num_classes = len(class_names)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # normalizes pixel values [0, 1]
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),  # downsamples
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# plots
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plot_model_metrics(epochs_range, acc, val_acc, loss, val_loss)

# loads test train_data
test_data_dir = pathlib.Path("test_data")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    batch_size=32,
    image_size=(img_width, img_height),
    seed=123,
    shuffle=False
)

# test_dataset visualization
show_images_from_set(test_ds, test_ds.class_names)

# evaluation has an average accuracy between 50-55 and loss of 2.1
# results = model.evaluate(test_ds)
# print(dict(zip(model.metrics_names, results)))

# indexing labels with class names, dataset is filled with images and labels
# by going through directories by order of A-Z
indexed_labels = {}
for image, label in test_ds:  # Eager Tensor
    for i in range(len(image)):
        # print(class_names[label[i]])
        indexed_labels[i] = class_names[label[i]]

# predicts values for 8 different classes
predictions = model.predict(test_ds)
for i, _ in enumerate(predictions):
    score = tf.nn.softmax(predictions[i])

    print(
        "This {} emoticon most likely belongs to {} with a {:.2f} percent confidence."
        .format(indexed_labels[i], class_names[np.argmax(score)], np.max(score) * 100)
    )

# Calculates evaluation metrics, plots confusion matrix
y_pred = np.argmax(predictions, axis=1)
y_test = np.concatenate([y for x, y in test_ds], axis=0)

print("Accuracy: ", accuracy_score(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, xticks_rotation="vertical",
                                        ax=ax, colorbar=False)

plt.show()
