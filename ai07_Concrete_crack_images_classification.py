# -*- coding: utf-8 -*-
"""
Created on Tue July  26 12:14:29 2022

@author: MAKMAL2-PC23
"""

#1. Import packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, datetime, pathlib

#%%
#2. Load images from file
SEED = 42
IMG_SIZE = (100,100)
BATCH_SIZE = 32

train_dataset = keras.utils.image_dataset_from_directory(r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\Concrete Crack Images for Classification",seed=SEED,image_size=IMG_SIZE,batch_size=BATCH_SIZE,subset='training',validation_split=0.3)
val_dataset = keras.utils.image_dataset_from_directory(r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\Concrete Crack Images for Classification",seed=SEED,image_size=IMG_SIZE,batch_size=BATCH_SIZE,subset='validation',validation_split=0.3)

#%%
#Further split validation into validation-test splits
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#Create prefetch dataset for all the 3 splits
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size = AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size = AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size = AUTOTUNE)

#%%
#3. Data augmentation layers
data_aug = tf.keras.Sequential()
data_aug.add(tf.keras.layers.RandomFlip())
data_aug.add(tf.keras.layers.RandomRotation(0.3))

#%%
#4. Create Feature Extraction layers & base model
feature_ext_layer = tf.keras.applications.resnet50.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)
class_names = train_dataset.class_names
base_model = tf.keras.applications.ResNet50(include_top = False, weights = "imagenet", input_shape = IMG_SHAPE)

base_model.trainable = False
base_model.summary()

#%%
#5. Create classification layers
avg_layer = tf.keras.layers.GlobalAveragePooling2D()
output_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")

#%%
#6. Create the entire model with Functional API
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_aug(inputs)
x = feature_ext_layer(x)
x = base_model(x, training=False)
x = avg_layer(x)
outputs = output_layer(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

#%%
#7. Compile the model
model.compile(optimizer= "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

#%%
#TensorBoard callback
base_log_path = r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_cb = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 2, min_delta = 0.2)
#Train the model
EPOCHS = 50
history = model.fit(pf_train, validation_data=pf_val, epochs=EPOCHS, callbacks=[tb_cb,es_cb])

#%%
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = history.epoch

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("Training vs Validation accuracy")
plt.legend()
plt.figure()

plt.show()

#%%
#Evaluate the model after training
loss, accuracy = model.evaluate(pf_test)

print(f"loss = {loss}")
print(f"Accuracy = {accuracy}")

#%%
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions,axis=1)

#%%
#8. Show some prediction results
plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"C:\Users\MAKMAL2-PC23\Documents\TensorFlow Deep Learning\Concrete Crack Images for Classification\result"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()