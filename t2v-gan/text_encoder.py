import torch
import torch.nn as nn

class TextEncoder(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_dim):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size,hidden_dim)
    self.text_projection=nn.Linear(hidden_dim,embedding_dim)
  def forward(self,inputs_ids,attention_mask=None):
    embedded=self.embedding(inputs_ids)
    if attention_mask is not None:
      mask=attention_mask.unsqueeze(-1).expand_as(embedded)
      embedded=embedded*mask
      sum_embeddings=embedded.sum(dim=1)
      num_tokens=attention_mask.sum(dim=1,keepdim=True).clamp(min=1)
      pooled_output=sum_embeddings/num_tokens
    else:
      pooled_output=torch.mean(embedded,dim=1)
    text_embeddings=self.text_projection(pooled_output)
    return text_embeddings