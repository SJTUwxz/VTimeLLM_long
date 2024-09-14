import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vtimellm_arch import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM
from torch.nn.utils.rnn import pad_sequence

def giou_1d_loss(prediction, groundtruth):
    # Ensure the start is always less than or equal to the end
    pred_start, pred_end = torch.min(prediction, dim=1)[0], torch.max(prediction, dim=1)[0]
    gt_start, gt_end = torch.min(groundtruth, dim=1)[0], torch.max(groundtruth, dim=1)[0]
    
    # Intersection: Maximum of the start points, minimum of the end points
    intersection_start = torch.max(pred_start, gt_start)
    intersection_end = torch.min(pred_end, gt_end)
    intersection = torch.clamp(intersection_end - intersection_start, min=0)
    
    # Union: Sum of individual lengths minus the intersection
    pred_length = pred_end - pred_start
    gt_length = gt_end - gt_start
    union = pred_length + gt_length - intersection
    
    # Enclosing segment: Smallest segment that can enclose both predicted and ground truth segments
    enclosing_start = torch.min(pred_start, gt_start)
    enclosing_end = torch.max(pred_end, gt_end)
    enclosing_length = enclosing_end - enclosing_start
    
    # IoU for 1D segments
    iou = intersection / union
    
    # GIoU for 1D segments
    giou = iou - (enclosing_length - union) / enclosing_length
    
    # GIoU Loss
    giou_loss = 1 - giou
    
    # Return the mean GIoU loss across all segments
    return giou_loss.mean()


class VTimeLLMConfig(LlamaConfig):
    model_type = "VTimeLLM"

class VTimeLLMLlamaModel(LlamaModel, VTimeLLMMetaModel):
    config_class = VTimeLLMConfig

    def __init__(self, config: LlamaConfig):
        super(VTimeLLMLlamaModel, self).__init__(config)

class VTimeLLMLlamaForCausalLM(LlamaForCausalLM, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VTimeLLMLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # TODO: segment_head should be added when temporal_loss in model arguments is set to True
        self.segment_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        segments: Optional[torch.FloatTensor] = None,
        segment_mask: Optional[torch.Tensor] = None,
        segment_indices: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        is_generate: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if self.get_model().model_args.temporal_loss and self.training:


            if inputs_embeds is None and not is_generate:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels, 
                    new_segment_indices,
                ) = self.prepare_inputs_labels_for_multimodal_temporal_loss(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    segments,
                    segment_indices,
                    segment_mask,
                )

            return_dict = True
            output_hidden_states = True
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_states = outputs.hidden_states[-1]
            selected_features = []
            for bs in range(last_hidden_states.shape[0]):
                output_segment_indices = list(map(lambda x: x - 1, new_segment_indices[bs]))
                selected_batch_features = last_hidden_states[bs, output_segment_indices, :]  # Shape: [seq_len, hidden_size]
                selected_features.append(selected_batch_features)
            selected_features = torch.cat(selected_features)
            segments_predictions = self.segment_head(selected_features).flatten()

            gt = []

            for bs in range(last_hidden_states.shape[0]):
                # Filter out the -1 elements from the i-th batch
                valid_elements = segments[bs][segments[bs] != -1]
                
                gt.extend(valid_elements)

            assert len(segments_predictions) == len(gt), "predicted segments should have the same length as groundtruth segments"

            gt = torch.tensor(gt).to(segments_predictions.device)

            if self.get_model().model_args.loss_type == "vanilla":
                loss = outputs.loss
            elif self.get_model().model_args.loss_type == "l1_loss":
                l1_loss = nn.L1Loss()(segments_predictions, gt)
                loss = outputs.loss + l1_loss 
            elif self.get_model().model_args.loss_type == "giou_loss":
                segments_predictions = segments_predictions.reshape(-1, 2)
                gt = gt.reshape(-1, 2)
                giou_loss = giou_1d_loss(segments_predictions, gt)
                loss = outputs.loss + giou_loss 
            elif self.get_model().model_args.loss_type == "l1_giou_loss":
                l1_loss = nn.L1Loss()(segments_predictions, gt)
                segments_predictions = segments_predictions.reshape(-1, 2)
                gt = gt.reshape(-1, 2)
                giou_loss = giou_1d_loss(segments_predictions, gt)
                loss = outputs.loss + l1_loss + giou_loss 
            
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        else:
            if inputs_embeds is None and not is_generate:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images
                )

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("VTimeLLM", VTimeLLMConfig)
AutoModelForCausalLM.register(VTimeLLMConfig, VTimeLLMLlamaForCausalLM)
