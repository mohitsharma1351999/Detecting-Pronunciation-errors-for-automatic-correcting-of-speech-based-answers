from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from data_utils import Data
import tensorflow as tf
from tensorflow.keras import models
from tensorflow import keras
import matplotlib.pyplot as plt
from model import MY_Model
from model0 import AudioEncoder
audio_encoder = AudioEncoder()
my_data = Data()
my_model = MY_Model()
epochs = 10


for e in range(epochs):         # loop for epoch
    # model load
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='my_model',
                                                     save_weights_only=True,
                                                     verbose=1)
    if e > 0:
        model.load_weights('my_model')
    for b in range(my_model.nbbatches):  # loop for batch
        yield_output = my_data.get_data_from_batches(
            my_data.xtrain, my_data.ytrain ,batch_size=10)
        xbatch = next(yield_output)
        ybatch = next(yield_output)
        print("jatt", xbatch.shape, ybatch.shape)
        required_width = next(yield_output)
        model = my_model.get_model()
        # input_shape = (required_width, 128)
        # xbatch should be B, local_max, 128
        # ybatch should be B, 100
        # model.train_on_batch(x=xbatch, y=ybatch
        #                      # sample_weight=None,
        #                      # class_weight=None,
        #                      # reset_metrics=True,
        #                      # return_dict=False,
        #                      )
        history = model.fit(xbatch, ybatch, epochs=10,
                            validation_data=0.25,
                            callbacks=[cp_callback])

    # model save
    model.save_weights('my_model'.format(epoch=0))
model.save('final_model/final_model.h5')

# loaded_model = tf.keras.models.load_model('final_model')
# for b in range(my_model.nbbatches):
#     yield_output1 = my_data.get_data_from_batches(
#         my_data.xtest, my_data.ytest, b, batch_size=10)
#     xbatch1 = next(yield_output1)
#     ybatch1 = next(yield_output1)
#     xbatch1 = np.array(xbatch1)
#     # Re-evaluate the model
#     test_loss, test_acc = loaded_model.evaluate(xbatch1, ybatch1, verbose=2)
#     print("Restored model, accuracy: {:5.2f}%".format(100 * test_acc))

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
