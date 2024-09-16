import torch
import torch.nn as nn
from vtimellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, TEMPORAL_TOKEN_INDEX
from abc import ABC, abstractmethod

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


    
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class VTimeLLMMetaModel:

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(768, self.config.hidden_size)

        # project temporal boundaries to tokens
        self.model_args = model_args

        if self.model_args.temporal_loss:
            if not hasattr(self, 'temporal_projector'):
                if self.model_args.projector_type == "simple_linear":
                    self.temporal_projector = nn.Linear(1, self.config.hidden_size)
                elif self.model_args.projector_type == "angular":
                    multires = 64
                    self.func = get_embedder(multires)
                    self.temporal_projector = nn.Linear(2*multires, self.config.hidden_size)


        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load mlp:", pretrain_mm_mlp_adapter)


class VTimeLLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal_temporal_loss(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, segments, segment_indices, segment_mask
    ):
        # print(position_ids, attention_mask)
        # if past_key_values:
        #     print(past_key_values[-1][-1].shape)
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                if self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().mm_projector(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.get_model().mm_projector(images)
        # print([image.shape for image in image_features])

        # max_seq_len: 
        # pad all sequences in segments to the maximum length 
        # [[0.0, 0.5], [0.1, 0.2, 0.3, 0.4]] -> [[0.0, 0.5, -1, -1], [0.1, 0.2, 0.3, 0.4]]
        #
        # segments shape: [batch, max_seq_len]
        segments = segments.unsqueeze(-1)
        # segments shape: [batch, max_seq_len, 1]
        # TODO: change linear to NERF-like fourier projection
        if self.get_model().model_args.projector_type == "simple_linear":
            temporal_features = self.get_model().temporal_projector(segments)
        elif self.get_model().model_args.projector_type == "angular":
            segments = segments.to(torch.bfloat16)
            temporal_features = self.get_model().func[0](segments)
            temporal_features = self.get_model().temporal_projector(temporal_features)
            # temporal_features = temporal_features.to(torch.float16)
            # segments = segments.to(torch.float16)
        # temporal_features shape: [batch, max_seq_len, hidden_size]


        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)


        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []

        new_segment_indices = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                raise NotImplementedError("segment is not implemented for num_images=0")
                cur_image_features = image_features[cur_image_idx]
                # curr_input_ids has -300 TEMPORAL_TOKEN_INDEX
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

            segment_indices_ = torch.where(cur_input_ids == TEMPORAL_TOKEN_INDEX)[0].tolist()

            num_segments = len(segment_indices_)
            segment_image_token_indices = image_token_indices + segment_indices_
            segment_image_token_indices.sort()

            cur_input_ids_noim_nosegment = []
            cur_labels = labels[batch_idx]
            cur_labels_noim_nosegment = []
            segment_or_image = []
            for i in range(len(segment_image_token_indices) - 1):

                if segment_image_token_indices[i] in image_token_indices:
                    if segment_image_token_indices[i] != -1:
                        segment_or_image.append(True)
                elif segment_image_token_indices[i] in segment_indices_:
                    segment_or_image.append(False)

                cur_input_ids_noim_nosegment.append(cur_input_ids[segment_image_token_indices[i]+1:segment_image_token_indices[i+1]])
                cur_labels_noim_nosegment.append(cur_labels[segment_image_token_indices[i]+1:segment_image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim_nosegment]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim_nosegment))
            cur_input_embeds_no_im_nosegment = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            cur_segment_idx = 0
            cur_segment_indices = []
            cur_embed_count = 0
            for i in range(num_images + num_segments + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im_nosegment[i])
                cur_new_labels.append(cur_labels_noim_nosegment[i])
                cur_embed_count += cur_input_embeds_no_im_nosegment[i].shape[0]
                if i < num_images + num_segments:
                    if segment_or_image[i]:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_embed_count += cur_image_features.shape[0]
                    else:
                        cur_segment_indices.append(cur_embed_count)
                        cur_embed_count += 1
                        cur_temporal_features = temporal_features[batch_idx, cur_segment_idx].unsqueeze(0)
                        cur_segment_idx += 1
                        cur_new_input_embeds.append(cur_temporal_features)
                        cur_new_labels.append(torch.full((cur_temporal_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            assert cur_segment_idx == num_segments, "number of segments should match number of temporal features"

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            
            new_segment_indices.append(cur_segment_indices)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000, 
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None
        # print(position_ids, attention_mask)
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_segment_indices

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # print(position_ids, attention_mask)
        # if past_key_values:
        #     print(past_key_values[-1][-1].shape)
        # print(input_ids.shape, position_ids.shape, attention_mask.shape, past_key_values.shape, images)
        # TODO: remove input_ids.shape[1] == 1 because of the evaluation forward function
        if images is None or input_ids.shape[1] == 1:
        # if images is None:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                if self.get_model().config.model_type == 'chatglm':
                    target_shape = past_key_values[-1][-1].shape[0] + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().mm_projector(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.get_model().mm_projector(images)
        # print([image.shape for image in image_features])
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if self.get_model().config.model_type == 'chatglm':
            fake_input_ids = torch.full((new_input_embeds.shape[0], new_input_embeds.shape[1]), -10000, 
                                        dtype=new_input_embeds.dtype, device=new_input_embeds.device)
            attention_mask = attention_mask.to(torch.int8)
            new_input_embeds = new_input_embeds.transpose(0, 1).contiguous()
        else:
            fake_input_ids = None
        # print(position_ids, attention_mask)
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
