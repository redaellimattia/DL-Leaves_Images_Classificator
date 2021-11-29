import pickle

import splitfolders
import tensorflow as tf
import numpy as np
import os
import random
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import visualkeras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
from datetime import datetime

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

labels = ['Apple',  # 0
          'Blueberry',  # 1
          'Cherry',  # 2
          'Corn',  # 3
          'Grape',  # 4
          'Orange',  # 5
          'Peach',  # 6
          'Pepper',  # 7
          'Potato',  # 8
          'Raspberry',  # 9
          'Soybean',  # 10
          'Squash',  # 11
          'Strawberry',  # 12
          'Tomato']  # 13

# DATASET FOLDER
data_dir = 'data'

# DATASET ALREADY UNZIPPED IN LOCAL

# Split the dataset into subfolders
dataset_dir = os.path.join(data_dir, 'dataset')
# Train : 70%, Val: 20%, Test: 10%
splitfolders.ratio(dataset_dir, output=data_dir, seed=1337, ratio=(0.7, 0.2, 0.1), group_prefix=None)

# SEED
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# METADATA
input_shape = (256, 256, 3)
batch_size = 32
epochs = 200

# LOAD DATA PERFORMING DATA AUGMENTATION SPECIFYING THE CORRECT PREPROCESS FUNCTION FOR RESNET
train_gen = ImageDataGenerator(rotation_range=30,
                               zoom_range=0.15,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.15,
                               horizontal_flip=True,
                               fill_mode="nearest",
                               preprocessing_function=preprocess_input).flow_from_directory(
    directory=os.path.join(data_dir, 'train'),
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,  # can be set to labels
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=os.path.join(data_dir, 'val'),
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
    seed=seed)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory=os.path.join(data_dir, 'test'),
    target_size=(256, 256),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
    seed=seed)

model = None
model_name = ""


def create_folders_and_callbacks():
    exps_dir = os.path.join('experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    model_dir = os.path.join(exps_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(model_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    # ----------------
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'),
                                                       save_weights_only=False,  # True to save only weights
                                                       save_best_only=False)  # True to save only the best epoch
    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    # ---------------------------------
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 profile_batch=0,
                                                 histogram_freq=1)  # if > 0 (epochs) shows weights histograms
    callbacks.append(tb_callback)

    # Early Stopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15,
                                                   restore_best_weights=True)

    # ReduceLROnPlateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1,
                                                     mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
    callbacks.append(es_callback)
    callbacks.append(reduce_lr)

    return callbacks


def plot_train_val(history):
    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'], label='Validation', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15, 5))
    plt.plot(history['accuracy'], label='Training', alpha=.8, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'], label='Validation', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()


def confusion_matrix_plot():
    # Confusion Matrix and Classification Report
    Y_pred = model.predict(test_gen, test_gen.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)
    print('Classification Report')
    print(classification_report(test_gen.classes, y_pred, target_names=labels))
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.T, xticklabels=labels, yticklabels=labels)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()


def supernet_util(supernet):
    # Setting up model starting with the supernet chosen in the init()
    global model
    supernet.trainable = False

    x = inputs = tfk.Input(shape=(256, 256, 3))
    x = supernet(x)
    x = tfkl.GlobalAveragePooling2D(name='GlobalAveragePooling')(x)
    x = tfkl.Dropout(0.8, seed=seed)(x)
    x = tfkl.Dense(
        256,
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)
    x = tfkl.Dropout(0.6, seed=seed)(x)
    outputs = tfkl.Dense(
        14,
        activation='softmax',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)

    model = tfk.Model(inputs=inputs, outputs=outputs, name='transfer_learning_model')

    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')


def fine_tuning(N, lr):
    global model
    print("Setting all layers to trainable... ")

    model.get_layer('resnet152v2').trainable = True
    for i, layer in enumerate(model.get_layer('resnet152v2').layers):
        print(i, layer.name, layer.trainable)

    # Freezing first N layers
    for i, layer in enumerate(model.get_layer('resnet152v2').layers[:N]):
        layer.trainable = False

    print("----LAYER STATUS----")
    for i, layer in enumerate(model.get_layer('resnet152v2').layers):
        print(i, layer.name, layer.trainable)

    # Compiling the model with lr as learning_rate
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(lr),
                  metrics='accuracy')


def training():
    print()
    print("TRAINING.....")
    # Create folders and callbacks and fit
    callbacks = create_folders_and_callbacks()

    # TRAINING
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
    ).history

    # PLOT TRAINING RESULTS
    plot_train_val(history)

    # SAVE THE HISTORY
    histories_dir = "histories"
    if not os.path.exists(histories_dir):
        os.makedirs(histories_dir)
    f = open(histories_dir + "/" + model_name + "_history", 'wb')
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # SAVE BEST EPOCH MODEL
    model.save("experiments/" + model_name + "/" + model_name + "_Best")
    print()
    print("----MODEL SAVED----")
    print()


def init():
    global model_name

    # SET UP TRANSFER LEARNING
    supernet = tfk.applications.resnet_v2.ResNet152V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    print("----SUPERNET----")
    supernet.summary()

    # Create the model with transfer learning
    supernet_util(supernet)

    visualkeras.layered_view(model, legend=True).show()

    print("----MODEL READY FOR TRAINING----")
    model.summary()

    model_name = "TRNSF_RESNET_FINAL"

    # Start training
    training()

    # FIRST FINE TUNING, FREEZING FIRST 540 LAYERS WITH LR = 1e-4
    fine_tuning(540, 1e-4)
    print("----MODEL READY FOR TRAINING|FINE TUNING----")
    model.summary()
    model_name += "_FT"

    # Start training
    training()

    # SECOND FINE TUNING, FREEZING FIRST 528 LAYERS WITH LR = 1e-4
    fine_tuning(528, 1e-5)
    print("----MODEL READY FOR TRAINING|FINE TUNING----")
    model.summary()
    model_name += "2"

    # Start training
    training()

    print()
    print("FINE TUNING DONE")
    # Plot the confusion matrix
    confusion_matrix_plot()


# Start execution
init()
