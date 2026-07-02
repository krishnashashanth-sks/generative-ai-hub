from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model

def build_model(input_shape,vocabulary):
    input_features = Input(shape=input_shape, name="input_features")
    x = Bidirectional(GRU(units=128, return_sequences=True, activation='tanh'), name="bidirectional_gru_1")(input_features)
    x = Dropout(0.2, name="dropout_1")(x)
    x = Bidirectional(GRU(units=128, return_sequences=True, activation='tanh'), name="bidirectional_gru_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    output_logits = Dense(len(vocabulary), name="output_logits")(x)
    model = Model(inputs=input_features, outputs=output_logits, name='asr_model_rnn_gru_ctc')
