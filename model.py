from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel,BertModel,RobertaModel, PreTrainedModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTContrastive(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.model = BertModel(config)
        self.W = nn.Linear(config.hidden_size,config.hidden_size)
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,trigger_mask=None,arg_mask=None,none_arg_mask=None,none_arg_length_mask=None):
        outputs =self.model(                 # [bs, length, emd] 
            input_ids,                      # [batch,length]
            attention_mask=attention_mask,  # padding
            token_type_ids=token_type_ids,  # sentence segmentation
            position_ids=position_ids,      # position emebedding
            head_mask=head_mask,            # ？
            inputs_embeds=inputs_embeds,    # lookup mat
        )[0]

        trigger_mask = torch.unsqueeze(trigger_mask, dim=2)                     #[bs,length, 1]
        trigger_reps = outputs * trigger_mask                                   #[bs,length, emd]
        trigger_reps = torch.sum(trigger_reps,dim=1)                            #[bs, emd]
        trigger_reps = trigger_reps/torch.sum(trigger_mask,dim=1)               #[bs, emd]

        arg_mask = torch.unsqueeze(arg_mask, dim=2)
        arg_reps = outputs * arg_mask
        arg_reps = torch.sum(arg_reps,dim=1)
        arg_reps = arg_reps/torch.sum(arg_mask,dim=1)

        none_arg_mask = torch.unsqueeze(none_arg_mask, dim=3) #[bs, max_contras_ent, length, 1]
        outputs_un = torch.unsqueeze(outputs,dim=1)           #[bs, 1, length, emd]
        none_arg_reps = outputs_un * none_arg_mask            #[bs, max_contras_ent, length, emd]
        none_arg_reps = torch.sum(none_arg_reps,dim=2)        #[bs, max_contras_ent, emd]
        none_arg_reps = none_arg_reps/(torch.sum(none_arg_mask,dim=2)+1e-8) #[bs, max_contras_ent, emd]

        pos_loss = torch.sum(self.W(trigger_reps) * arg_reps,dim=1)          #[bs]

        neg_loss_1 = torch.mm(self.W(trigger_reps), torch.transpose(arg_reps,0,1))  #[bs,bs]
        neg_loss_1 = torch.exp(neg_loss_1)
        neg_loss_1 = torch.sum(neg_loss_1,dim=1)                             #[bs]

        trigger_reps = torch.unsqueeze(trigger_reps,dim=1)
        neg_loss_2 = torch.sum(self.W(trigger_reps) * none_arg_reps, dim=2) #[bs, max_contras_ent]
        neg_loss_2 = torch.exp(neg_loss_2) * none_arg_length_mask
        neg_loss_2 = torch.sum(neg_loss_2,dim=1)                            #[bs]

        loss = -pos_loss + torch.log(neg_loss_2+neg_loss_1+1e-8)

        loss = torch.mean(loss)
        return (loss,)


class DMBERT(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.maxpooling=nn.MaxPool1d(128) #???????
        self.classifier=nn.Linear(config.hidden_size*2,config.num_labels)
        self.W = nn.Linear(config.hidden_size,config.hidden_size)
    
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, maskL=None, maskR=None, labels=None):
        batchSize=input_ids.size(0)
        outputs =self.bert(
            input_ids,                      # [batch,length]
            attention_mask=attention_mask,  # padding
            token_type_ids=token_type_ids,  # sentence segmentation
            position_ids=position_ids,      # position emebedding
            head_mask=head_mask,            # ？
            inputs_embeds=inputs_embeds,    # lookup mat
        )
        conved=outputs[0]
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        R=R+torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(batchSize,self.config.hidden_size)
        pooledR=self.maxpooling(R).contiguous().view(batchSize,self.config.hidden_size)
        pooled=torch.cat((pooledL,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
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


class DMRoBERTa(PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert=RobertaModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.maxpooling=nn.MaxPool1d(128) 
        self.classifier=nn.Linear(config.hidden_size*2,config.num_labels)
        self.W = nn.Linear(config.hidden_size,config.hidden_size)
    
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, maskL=None, maskR=None, labels=None):
        batchSize=input_ids.size(0)
        outputs =self.bert(
            input_ids,                      # [batch,length]
            attention_mask=attention_mask,  # padding
            token_type_ids=token_type_ids,  # sentence segmentation
            position_ids=position_ids,      # position emebedding
            head_mask=head_mask,            # ？
            inputs_embeds=inputs_embeds,    # lookup mat
        )
        conved=outputs[0]
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        R=R+torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(batchSize,self.config.hidden_size)
        pooledR=self.maxpooling(R).contiguous().view(batchSize,self.config.hidden_size)
        pooled=torch.cat((pooledL,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
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


class RoBERTaContrastive(PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.model = RobertaModel(config)
        self.W = nn.Linear(config.hidden_size,config.hidden_size)
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,trigger_mask=None,arg_mask=None,none_arg_mask=None,none_arg_length_mask=None):
        outputs =self.model(                 # [bs, length, emd] 
            input_ids,                      # [batch,length]
            attention_mask=attention_mask,  # padding
            token_type_ids=token_type_ids,  # sentence segmentation
            position_ids=position_ids,      # position emebedding
            head_mask=head_mask,            # ？
            inputs_embeds=inputs_embeds,    # lookup mat
        )[0]

        trigger_mask = torch.unsqueeze(trigger_mask, dim=2)                     #[bs,length, 1]
        trigger_reps = outputs * trigger_mask                                   #[bs,length, emd]
        trigger_reps = torch.sum(trigger_reps,dim=1)                            #[bs, emd]
        trigger_reps = trigger_reps/torch.sum(trigger_mask,dim=1)               #[bs, emd]

        arg_mask = torch.unsqueeze(arg_mask, dim=2)
        arg_reps = outputs * arg_mask
        arg_reps = torch.sum(arg_reps,dim=1)
        arg_reps = arg_reps/torch.sum(arg_mask,dim=1)

        none_arg_mask = torch.unsqueeze(none_arg_mask, dim=3) #[bs, max_contras_ent, length, 1]
        outputs_un = torch.unsqueeze(outputs,dim=1)           #[bs, 1, length, emd]
        none_arg_reps = outputs_un * none_arg_mask            #[bs, max_contras_ent, length, emd]
        none_arg_reps = torch.sum(none_arg_reps,dim=2)        #[bs, max_contras_ent, emd]
        none_arg_reps = none_arg_reps/(torch.sum(none_arg_mask,dim=2)+1e-8) #[bs, max_contras_ent, emd]

        pos_loss = torch.sum(self.W(trigger_reps) * arg_reps,dim=1)          #[bs]

        neg_loss_1 = torch.mm(self.W(trigger_reps), torch.transpose(arg_reps,0,1))  #[bs,bs]
        neg_loss_1 = torch.exp(neg_loss_1)
        neg_loss_1 = torch.sum(neg_loss_1,dim=1)                             #[bs]

        trigger_reps = torch.unsqueeze(trigger_reps,dim=1)
        neg_loss_2 = torch.sum(self.W(trigger_reps) * none_arg_reps, dim=2) #[bs, max_contras_ent]
        neg_loss_2 = torch.exp(neg_loss_2) * none_arg_length_mask
        neg_loss_2 = torch.sum(neg_loss_2,dim=1)                            #[bs]

        loss = -pos_loss + torch.log(neg_loss_2+neg_loss_1+1e-8)

        loss = torch.mean(loss)
        return (loss,)

