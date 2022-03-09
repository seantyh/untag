'''
Source Code:
https://github.com/THUDM/P-tuning-v2/blob/main/model/multiple_choice.py#L467

'''


import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


class BertPromptForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config, custom_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.numchoices = custom_config['numchoices']
        self.bert = BertModel(config)
        self.embeddings = self.bert.embeddings
        self.n_class = custom_config['n_class']
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        for param in self.bert.parameters():
            param.requires_grad = custom_config['train_bert']
        ####
        self.pre_seq_len = custom_config['n_tokens']

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.n_class,
                                self.pre_seq_len * config.hidden_size)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('** total param is {}'.format(total_param)) 
        print('** train bert? {}'.format(custom_config['train_bert'])) 

    def get_prompt(self, seq_classes):
        '''
        type:
        seq_classes: toch.tensor
        rtype:

        '''
        # seqclasses:
        #  [[[11], [13]],
        # [[6], [7]],
        # [[4], [17]]...]
        # prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        batch_size = seq_classes.shape[0]
        prompts = self.prefix_encoder(seq_classes.squeeze())
        prompts =  prompts.reshape((batch_size*self.numchoices, -1, self.config.hidden_size))
        # prompts.shape = (batch_size*numchoice, prompt_len, hidden_size)
        # print('prompts:', prompts.shape)
        return prompts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_selector = None
    ):
        ###
        seq_classes = class_selector
        ####
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds[:2]

        input_ids = input_ids.reshape(-1, input_ids.size(-1)) if input_ids is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # print('input ids:', input_ids.shape)

        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        # type_shape = self.embeddings.token_type_ids.shape[0]
        # self.embeddings.token_type_ids = torch.cat([torch.ones([type_shape, self.pre_seq_len]), 
        #                                 self.embeddings.token_type_ids], dim = 1)
        # self.embeddings.token_type_ids = self.embeddings.token_type_ids.type(torch.LongTensor)
        prompts = self.get_prompt(
            seq_classes = seq_classes)
        
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        

        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # print('attention_mask shape:', attention_mask.shape)
        outputs = self.bert(
            input_ids = None, 
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
