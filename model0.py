import keras
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
import librosa
import time
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from data_utils import Data
label_encoder = LabelEncoder()
my_data = Data()
class AudioEncoder:
    def __init__(self,cnn_filters,rnn_nodes,nc):
        input_layer = keras.layers.Input(shape=(None,128))
        cnn_1 = keras.layers.Conv1D(cnn_filters[0],3,data_format="channels_last")(input_layer)
        cnn_2 = keras.layers.Conv1D(cnn_filters[1],3,data_format="channels_last")(cnn_1)
        rnn_1 = keras.layers.GRU(rnn_nodes[0],activation="relu",return_sequences=True,  kernel_initializer='glorot_uniform')(cnn_2)
        rnn_2 = keras.layers.GRU(rnn_nodes[1],activation="relu",return_sequences=False)(rnn_1) # encoding layer
        output_layer = keras.layers.Dense(nc,activation='softmax')(rnn_2)
        self.encoder = keras.Model(input_layer,rnn_2)
        self.classifier = keras.Model(input_layer,output_layer)
        self.classifier.compile(optimizer=Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    
    def train_classifier(self,X,Y,batch_size=32,epochs=500):
        # should be a generator
        # yields batch_x [N,None,F] padded mfcc array
        # yields batch_y [N,nc] one hot encoding
        batch_loss_list = []
        nbbatches = int(np.ceil(len(X)/batch_size))
        for e in range(epochs):
            if e > 0:
                self.classifier.load_weights('my_model')
            e_loss = 0
            e_acc = 0
            batches = my_data.get_data_from_batches(X,Y,batch_size)
            for b in range(nbbatches): # should compute prior
                batch_x, batch_y = next(batches)
                # print(batch_x.shape)
                # print(batch_y.shape)
                [batch_loss,batch_acc] = self.classifier.train_on_batch(batch_x,batch_y)
                batch_loss_list.append(batch_loss)
                e_loss += batch_loss
                e_acc += batch_acc
            e_loss /= nbbatches
            e_acc /= nbbatches
            # Plot history: MAE
            plt.plot(batch_loss_list, label='loss')
            # plt.plot(e, label='epochs (training data)')
            print("aversge train accuracy---->", e_acc, "final batch train accuracy -->", batch_acc)
            print("Epoch %d/%d Loss %0.4f"%(e,epochs,e_loss))
            self.classifier.save_weights('my_model'.format(epoch=0))
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('No. epoch')
        plt.show()
        self.classifier.save('my_model/model.h5')
    def test_classifier(self,X,Y,batch_size=32):
        print('entered the testing')
        nbbatches = int(np.ceil(len(X)/batch_size))
        e_loss = 0
        e_acc = 0
        batches = my_data.get_data_from_batches(X,Y,batch_size)
        for b in range(nbbatches): # should compute prior
            batch_x, batch_y = next(batches)
            [batch_loss,batch_acc] = self.classifier.test_on_batch(batch_x,batch_y)
            e_loss += batch_loss
            e_acc += batch_acc
        e_loss /= nbbatches
        e_acc /= nbbatches
        print("average test accuracy---->", e_acc, "final batch test accuracy -->", batch_acc)
        print("Loss %0.4f"%(e_loss))
    def prediction(self, file_path):
        y, sr = librosa.load(file_path)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel = np.transpose(mel, [1, 0])
        y_pred = self.classifier.predict(mel)
        inverted = label_encoder.inverse_transform([argmax(y_pred)])
        print('predicted output---->', inverted)
        return inverted


# data = np.random.random([2,17,40])
network = AudioEncoder([16,32],[64,128],100)
network.encoder.summary()
network.classifier.summary()
# output = network.encoder(data)
# print(output.shape)

train_x = my_data.xtrain
train_y = my_data.ytrain
start_time = time.time()
network.train_classifier(train_x, train_y, 32)
test_x = my_data.xtest
test_y = my_data.ytest
network.test_classifier(test_x, test_y, 16)
end_time = time.time()
print('total time taken', end_time-start_time)