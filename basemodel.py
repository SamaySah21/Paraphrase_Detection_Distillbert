import config
import transformers
import torch.nn as nn
import torch   
        
class DistilBertModel(nn.Module):
    def __init__(self, bert_path):
        super(DistilBertModel, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.DistilBertModel.from_pretrained(self.bert_path, output_hidden_states=True)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)


    def forward(self, ids):
        distilbert_output = self.bert(input_ids = ids)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled = self.dropout(pooled_output) 
        return self.out(pooled)

