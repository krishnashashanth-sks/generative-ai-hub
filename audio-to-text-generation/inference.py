import tensorflow as tf

def decode_batch_predictions(pred_batches, int_to_char_map, blank_token_int):
    results = []
    for pred_batch in pred_batches:
        # pred_batch shape: (batch_size, timesteps, num_classes)

        # Calculate input_length for CTC decoder
        # This is the actual length of each sequence in the batch (number of timesteps)
        input_len_tf = tf.cast(tf.fill([tf.shape(pred_batch)[0]], tf.shape(pred_batch)[1]), dtype=tf.int32)

        # Transpose pred_batch to be time-major: (timesteps, batch_size, num_classes)
        time_major_predictions = tf.transpose(pred_batch, perm=[1, 0, 2])

        # Use tf.nn.ctc_greedy_decoder (or tf.nn.ctc_beam_search_decoder)
        # It expects log-probabilities, so apply tf.math.log
        decoded, log_prob = tf.nn.ctc_greedy_decoder(
            inputs=tf.math.log(time_major_predictions + tf.keras.backend.epsilon()),
            sequence_length=input_len_tf
        )

        # Convert the SparseTensor output (decoded[0]) to a dense tensor for easier processing
        decoded_dense = tf.sparse.to_dense(decoded[0])

        for i in range(tf.shape(decoded_dense)[0]): # Iterate over items in the batch
            # Convert dense tensor for this batch item to numpy and filter out blank tokens
            text = decoded_dense[i].numpy()
            text = [int_to_char_map[t] for t in text if t != blank_token_int]
            results.append("".join(text))
    return results
