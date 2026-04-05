from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import ModelOutput

@dataclass
class MultiTaskSentimentOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    polarity_logits: Optional[torch.FloatTensor] = None
    emotion_logits: Optional[torch.FloatTensor] = None
    tone_logits: Optional[torch.FloatTensor] = None
    intensity_pred: Optional[torch.FloatTensor] = None
    token_tone_logits: Optional[torch.FloatTensor] = None
    token_emotion_logits: Optional[torch.FloatTensor] = None


class MultiTaskBertForSentiment(BertPreTrainedModel):
    def __init__(
        self,
        config,
        num_polarity_labels=3,
        num_emotion_labels=7,
        num_tone_labels=3,
        polarity_loss_weight=1.0,
        emotion_loss_weight=1.0,
        tone_loss_weight=1.0,
        intensity_loss_weight=1.0,
        token_tone_loss_weight=0.5,
        token_emotion_loss_weight=0.5,
    ):
        super().__init__(config)

        self.num_polarity_labels = num_polarity_labels
        self.num_emotion_labels = num_emotion_labels
        self.num_tone_labels = num_tone_labels

        self.polarity_loss_weight = polarity_loss_weight
        self.emotion_loss_weight = emotion_loss_weight
        self.tone_loss_weight = tone_loss_weight
        self.intensity_loss_weight = intensity_loss_weight
        self.token_tone_loss_weight = token_tone_loss_weight
        self.token_emotion_loss_weight = token_emotion_loss_weight

        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        self.polarity_head = nn.Linear(hidden_size, num_polarity_labels)
        self.emotion_head = nn.Linear(hidden_size, num_emotion_labels)
        self.tone_head = nn.Linear(hidden_size, num_tone_labels)
        self.intensity_head = nn.Linear(hidden_size, 1)

        self.token_tone_head = nn.Linear(hidden_size, num_tone_labels)
        self.token_emotion_head = nn.Linear(hidden_size, num_emotion_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        polarity_labels=None,
        emotion_labels=None,
        tone_labels=None,
        intensity_labels=None,
        token_tone_labels=None,
        token_emotion_labels=None,
        return_dict=True,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if pooled_output is None:
            pooled_output = sequence_output[:, 0]

        seq_repr = self.dropout(pooled_output)
        token_repr = self.dropout(sequence_output)

        polarity_logits = self.polarity_head(seq_repr)
        emotion_logits = self.emotion_head(seq_repr)
        tone_logits = self.tone_head(seq_repr)
        intensity_pred = self.intensity_head(seq_repr).squeeze(-1)

        token_tone_logits = self.token_tone_head(token_repr)
        token_emotion_logits = self.token_emotion_head(token_repr)

        total_loss = None
        ce_loss = nn.CrossEntropyLoss()
        token_ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        mse_loss = nn.MSELoss()

        losses = []

        if polarity_labels is not None:
            losses.append(self.polarity_loss_weight * ce_loss(polarity_logits, polarity_labels))

        if emotion_labels is not None:
            losses.append(self.emotion_loss_weight * ce_loss(emotion_logits, emotion_labels))

        if tone_labels is not None:
            losses.append(self.tone_loss_weight * ce_loss(tone_logits, tone_labels))

        if intensity_labels is not None:
            intensity_labels = intensity_labels.float()
            losses.append(self.intensity_loss_weight * mse_loss(intensity_pred, intensity_labels))

        if token_tone_labels is not None:
            token_tone_loss = token_ce_loss(
                token_tone_logits.view(-1, self.num_tone_labels),
                token_tone_labels.view(-1),
            )
            losses.append(self.token_tone_loss_weight * token_tone_loss)

        if token_emotion_labels is not None:
            token_emotion_loss = token_ce_loss(
                token_emotion_logits.view(-1, self.num_emotion_labels),
                token_emotion_labels.view(-1),
            )
            losses.append(self.token_emotion_loss_weight * token_emotion_loss)

        if len(losses) > 0:
            total_loss = sum(losses)

        if not return_dict:
            output = (
                polarity_logits,
                emotion_logits,
                tone_logits,
                intensity_pred,
                token_tone_logits,
                token_emotion_logits,
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskSentimentOutput(
            loss=total_loss,
            polarity_logits=polarity_logits,
            emotion_logits=emotion_logits,
            tone_logits=tone_logits,
            intensity_pred=intensity_pred,
            token_tone_logits=token_tone_logits,
            token_emotion_logits=token_emotion_logits,
        )