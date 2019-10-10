import pytorch_transformers
from pytorch_transformers import GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel,AdamW, WEIGHTS_NAME, CONFIG_NAME
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class GPT2ClassHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2ClassHeadsModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        
        self.classifier = nn.Linear(config.n_embd, 2)
        # self.classifier = nn.Sequential(nn.Linear(config.n_embd, 768), nn.ReLU(), nn.Dropout(p=0.2),
        #                                 nn.Linear(768, 2))
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()


    def forward(self, input_ids, labels=None, token_type_ids=None,
                position_ids=None, past=None, head_mask=None):
        transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                               past=past, head_mask=head_mask)
        hidden_states = transformer_outputs[0] # torch.Size([1, 124, 1024])
        logits = self.classifier(hidden_states[:,-1,:]) # torch.Size([1,2])
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits,labels)
        
        return loss,logits
