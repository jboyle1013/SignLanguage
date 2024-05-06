import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, TimeDistributed, Concatenate, Dropout
from tensorflow.keras.applications import VGG16
from spektral.layers import GCNConv
from tcn import TCN
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import backend as K


class GraphConvolution(Layer):
    def __init__(self, units, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs, adj):
        xw = K.dot(inputs, self.w)
        out = K.dot(adj, xw)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


class SignLanguageModel:
    def __init__(self, cnn_input_shape, hand_num_nodes, body_num_nodes, features_per_node, timesteps):
        self.cnn_input_shape = cnn_input_shape
        self.hand_num_nodes = hand_num_nodes
        self.body_num_nodes = body_num_nodes
        self.features_per_node = features_per_node
        self.timesteps = timesteps
        self.model = self.build_model()

    def create_cnn(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.cnn_input_shape)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        return Model(inputs=base_model.input, outputs=x)

    def create_gcn(self, num_nodes):
        inputs = tf.keras.Input(shape=(num_nodes, self.features_per_node))
        adj_input = tf.keras.Input(shape=(num_nodes, self.features_per_node))
        gc1 = GraphConvolution(64)(inputs, adj_input)
        return Model(inputs=[inputs, adj_input], outputs=gc1)

    def create_tcn(self, input_shape):
        inp = Input(shape=input_shape)
        x = TCN(nb_filters=64, kernel_size=3, nb_stacks=1, dilations=[1, 2, 4, 8], padding='causal')(inp)
        out = Dense(10, activation='softmax')(x)
        return Model(inputs=inp, outputs=out)

    def build_model(self):
        # cnn = self.create_cnn()
        gcn_body = self.create_gcn(self.body_num_nodes)
        gcn_hand_left = self.create_gcn(self.hand_num_nodes)
        gcn_hand_right = self.create_gcn(self.hand_num_nodes)
        combined_features_shape = (self.timesteps)

        video_input = Input(shape=(self.timesteps,) + self.cnn_input_shape)
        adj_input_body = Input(shape=(self.body_num_nodes, self.body_num_nodes))
        adj_input_hand = Input(shape=(self.hand_num_nodes, self.hand_num_nodes))

        # cnn_out = TimeDistributed(cnn)(video_input)
        gcn_body_out = TimeDistributed(gcn_body)(video_input, adj_input_body)
        gcn_hand_left_out = TimeDistributed(gcn_hand_left)(video_input, adj_input_hand)
        gcn_hand_right_out = TimeDistributed(gcn_hand_right)(video_input, adj_input_hand)

        all_features = Concatenate()([ gcn_body_out, gcn_hand_left_out, gcn_hand_right_out])
        tcn = self.create_tcn(combined_features_shape)
        final_output = tcn(all_features)

        return Model(inputs=[video_input, adj_input_body, adj_input_hand], outputs=final_output)


    def train(self, data, labels, batch_size, epochs):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(data, labels, batch_size=batch_size, epochs=epochs)

    def predict(self, data):
        return self.model.predict(data)


if __name__ == '__main__':

    model = SignLanguageModel(cnn_input_shape=(128, 128, 3), hand_num_nodes=21, body_num_nodes=25, features_per_node=2,
                              timesteps=64)
    model.create_gcn(21)
    # model.build_model()
    #
    # model.summary()