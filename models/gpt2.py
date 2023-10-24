import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                                position_ids=position_ids, head_mask=head_mask,
                                                inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs