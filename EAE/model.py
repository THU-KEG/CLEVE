from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel,RobertaModel, BertModel

class DMBERT(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.roberta=BertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.maxpooling=nn.MaxPool1d(128) #???????
        self.classifier=nn.Linear(config.hidden_size*3,config.num_labels)
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, maskL=None, maskM=None, maskR=None, labels=None):
        batchSize=input_ids.size(0)
        outputs =self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        conved=outputs[0]
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        M=(conved*maskM).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+10*torch.ones_like(L)
        M=M+10*torch.ones_like(M)
        R=R+10*torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(batchSize,self.config.hidden_size)
        pooledM=self.maxpooling(M).contiguous().view(batchSize,self.config.hidden_size)
        pooledR=self.maxpooling(R).contiguous().view(batchSize,self.config.hidden_size)
        pooled=torch.cat((pooledL,pooledM,pooledR),1)
        pooled=pooled-10*torch.ones_like(pooled)
        pooled=self.dropout(pooled)
        logits=self.classifier(pooled)
        reshaped_logits=logits.view(-1, self.config.num_labels)
        outputs = (reshaped_logits,) + outputs[2:]
        #rep=torch.cat((pooled,loc_embeds),1)
        #rep=F.tanh(self.dropout(pooled))
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            loss=loss_fct(reshaped_logits, labels)
            outputs=(loss,)+outputs
        return outputs


class DMRoBERTa(BertPreTrainedModel):
    def __init__(self,config, bert):
        super().__init__(config)
        self.bert=bert
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.maxpooling=nn.MaxPool1d(128) #???????
        self.classifier=nn.Linear(config.hidden_size*3,config.num_labels)
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, maskL=None, maskM=None, maskR=None, labels=None):
        batchSize=input_ids.size(0)
        outputs =self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        conved=outputs[0]
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        M=(conved*maskM).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+10*torch.ones_like(L)
        M=M+10*torch.ones_like(M)
        R=R+10*torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(batchSize,self.config.hidden_size)
        pooledM=self.maxpooling(M).contiguous().view(batchSize,self.config.hidden_size)
        pooledR=self.maxpooling(R).contiguous().view(batchSize,self.config.hidden_size)
        pooled=torch.cat((pooledL,pooledM,pooledR),1)
        pooled=pooled-10*torch.ones_like(pooled)
        pooled=self.dropout(pooled)
        logits=self.classifier(pooled)
        reshaped_logits=logits.view(-1, self.config.num_labels)
        outputs = (reshaped_logits,) + outputs[2:]
        #rep=torch.cat((pooled,loc_embeds),1)
        #rep=F.tanh(self.dropout(pooled))
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            loss=loss_fct(reshaped_logits, labels)
            outputs=(loss,)+outputs
        return outputs