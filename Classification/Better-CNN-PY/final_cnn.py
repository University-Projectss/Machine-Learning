import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# citirea datelor de antrenare si validare
train_data = pd.read_csv("/kaggle/input/unibuc-dhc-2023/train.csv")
train_images_name = np.array(train_data['Image'])
train_labels = np.array(train_data['Class'])

val_data = pd.read_csv("/kaggle/input/unibuc-dhc-2023/val.csv")
val_images_name = np.array(val_data['Image'])
val_labels = np.array(val_data['Class'])


# prelucrarea datelor de antrenare
train_images = [np.asarray(Image.open(f'/kaggle/input/unibuc-dhc-2023/train_images/{img}')) / 255.0 for img in train_images_name]
val_images = [np.asarray(Image.open(f'/kaggle/input/unibuc-dhc-2023/val_images/{img}')) / 255.0 for img in val_images_name]


flipX = np.array([tf.image.per_image_standardization(i)
                 for i in tf.image.flip_left_right(train_images)])
flipY = np.array([tf.image.per_image_standardization(i)
                 for i in tf.image.flip_up_down(train_images)])
train_images = np.array([tf.image.per_image_standardization(i)
                        for i in train_images])

train_images = np.append(train_images, flipX, axis=0)
train_images = np.append(train_images, flipY, axis=0)

aux = train_labels
train_labels = np.append(train_labels, aux)
train_labels = np.append(train_labels, aux)
train_labels = np.array([[i] for i in train_labels])


val_images = np.array([tf.image.per_image_standardization(i)
                      for i in val_images])

val_labels = np.array([[i] for i in val_labels])


# initializarea modelului
window_size = 3
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=80, kernel_size=5,
          activation='relu', input_shape=(64, 64, 3)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(
    filters=256, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(tf.keras.layers.Conv2D(
    filters=176, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(
    filters=176, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(
    filters=200, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(
    filters=600, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.GlobalAveragePooling2D())


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(96, activation='relu',
          bias_initializer=tf.keras.initializers.glorot_uniform))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(rate=0.5))

model.add(tf.keras.layers.Dense(96, activation='softmax'))

model.summary()


# procesul de antrenare si afisare a graficului din timpul antrenarii

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy']
              )

# 0.007 -> 83%
# 0.005 -> 84%
# 0.001 -> 87%
model.optimizer.learning_rate = 0.0005

results = model.fit(x=train_images,
                    y=np.array([i for i in train_labels]),
                    epochs=50,
                    validation_data=(val_images, val_labels),
                    shuffle=True,
                    batch_size=80,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
                               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2)])

test_loss, test_acc = model.evaluate(val_images,  val_labels, verbose=2)

plt.plot(results.history['accuracy'], label='train_accuracy')
plt.plot(results.history['val_accuracy'], label='val_accuracy')
plt.plot(results.history['val_loss'], label='val_loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='upper left')

print("Score: ", test_acc)


# scrierea predictiilor pe datele de testare
test_data = pd.read_csv("/kaggle/input/unibuc-dhc-2023/test.csv")['Image']

images = [np.asarray(Image.open(f'/kaggle/input/unibuc-dhc-2023/test_images/{img}')) / 255.0 for img in test_data]

images = np.array([tf.image.per_image_standardization(i) for i in images])

images = np.array(images)
# print("TEST IMAGES", images)

predictions = model.predict(images)
# print(predictions.shape)
# print("Predictions: ", predictions[0])

data = {'Image': test_data, 'Class': [np.argmax(i) for i in predictions]}
pd.DataFrame(data).to_csv('/kaggle/working/sample_submission.csv', index=False)
print("FINISH!")


# afisarea matricei de confuzie
# predictions = model.predict(val_images)
# conf_matrix = confusion_matrix(
#     val_labels, [np.argmax(i) for i in predictions], labels=[i for i in range(97)])

# plt.imshow(conf_matrix)
# plt.show()

# afisam classification report pentru foecare clasa
# print(classification_report(val_labels, [np.argmax(i) for i in predictions]))
