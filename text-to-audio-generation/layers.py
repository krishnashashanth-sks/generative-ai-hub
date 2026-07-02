import torch.nn.functional as F
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
  def __init__(self,num_embeddings,embedding_dim,hidden_size,num_layers,dropout_prob,bidirectional):
    super().__init__()
    self.embedding=nn.Embedding(num_embeddings,embedding_dim)
    self.lstm=nn.LSTM(
        embedding_dim,
        hidden_size,
        num_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=dropout_prob if num_layers>1 else 0
    )
    self.hidden_size=hidden_size
    self.num_layers=num_layers
    self.bidirectional=bidirectional
  def forward(self,text_sequences,input_lengths):
    embedded_sequences=self.embedding(text_sequences)
    input_lengths_cpu=input_lengths.cpu().long()
    packed_embedded=nn.utils.rnn.pack_padded_sequence(
        embedded_sequences,input_lengths_cpu,batch_first=True,enforce_sorted=False
    )
    packed_lstm_out,(hidden_cell)=self.lstm(packed_embedded)
    lstm_out,_=nn.utils.rnn.pad_packed_sequence(packed_lstm_out,batch_first=True)
    return lstm_out
  
class AttentionMechanism(nn.Module):
  def __init__(self,encoder_output_dim,decoder_hidden_dim,attention_dim):
    super().__init__()
    self.W1=nn.Linear(encoder_output_dim,attention_dim)
    self.W2=nn.Linear(decoder_hidden_dim,attention_dim)
    self.V=nn.Linear(attention_dim,1)
  def forward(self,encoder_outputs,decoder_hidden):
    decoded_hidden_expanded=decoder_hidden.unsqueeze(1).expand(-1,encoder_outputs.size(1),-1)
    energy=torch.tanh(self.W1(encoder_outputs)+self.W2(decoded_hidden_expanded))
    attention_energies=self.V(energy)
    attention_weights = F.softmax(attention_energies.squeeze(2), dim=1)
    context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
    return context_vector,attention_weights
  
class AudioDecoder(nn.Module):
  def __init__(self,input_dim,hidden_size,num_layers,output_dim,dropout_prob=0.0):
    super().__init__()
    self.hidden_size=hidden_size
    self.num_layers=num_layers
    self.output_dim=output_dim
    self.lstm_cell=nn.LSTMCell(input_dim,hidden_size)
    self.output_linear=nn.Linear(hidden_size,output_dim)
  def forward(self,context_vector,prev_mel_frame,decoder_hidden,decoder_cell):
    """
        Performs a single decoding step.

        Args:
            context_vector (torch.Tensor): Context vector from the attention mechanism.
                                          Shape: (batch_size, encoder_output_dim)
            prev_mel_frame (torch.Tensor): Previous Mel-spectrogram frame (or start token).
                                          Shape: (batch_size, n_mels)
            decoder_hidden (torch.Tensor): Previous decoder hidden state.
                                          Shape: (batch_size, hidden_size)
            decoder_cell (torch.Tensor): Previous decoder cell state.
                                        Shape: (batch_size, hidden_size)

        Returns:
            tuple: A tuple containing:
                - predicted_mel_frame (torch.Tensor): The predicted next Mel-spectrogram frame.
                                                      Shape: (batch_size, n_mels)
                - new_decoder_hidden (torch.Tensor): The new decoder hidden state.
                                                   Shape: (batch_size, hidden_size)
                - new_decoder_cell (torch.Tensor): The new decoder cell state.
                                                  Shape: (batch_size, hidden_size)
        """
    lstm_input=torch.cat((context_vector,prev_mel_frame),dim=1)
    new_decoder_hidden,new_decoder_cell=self.lstm_cell(lstm_input,(decoder_hidden,decoder_cell))
    predicted_mel_frame=self.output_linear(new_decoder_hidden)
    return predicted_mel_frame,new_decoder_hidden,new_decoder_cell