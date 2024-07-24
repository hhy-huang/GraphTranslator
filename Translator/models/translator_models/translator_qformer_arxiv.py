"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from common.registry import registry
from models.base_model import all_gather_with_grad, concat_all_gather
from models.translator_models.translator import (
    TranslatorBase,
    compute_sim_matrix
)
from models.translator_models.translator_outputs import TranslatorOutput, TranslatorOutputFeatures

from transformers import BertTokenizer
from models.translator_models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("translator_arxiv")
class TranslatorQformerArxiv(TranslatorBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_arxiv": "train/pretrain_arxiv_stage1.yaml",
        "translator_generate_stage1": "train/pretrain_socialgenerate_stage1.yaml",
    }

    def __init__(
        self,
        num_features=768,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.use_neighbors = False
        self.GNN_embeddings = torch.load("/data/ChenWei/HaoyuHuang/GraphTranslator/data/arxiv/GraphTranslator-arxiv/graphsage_node_embeddings.pt").to('cpu').detach().numpy()
        self.tokenizer = self.init_tokenizer()                      # tokenizer from BERT, add [DEC] special token

        self.Qformer, self.query_tokens = self.init_Qformer(        # init Qformer
            num_query_token, num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.behavior_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)  # [768, 256]
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)      # [768, 256]

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)               # [768, 2]

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len                                              # 512
        self.proj = nn.Sequential(                                                  # Projector, [256, 256]
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("../models/bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("../models/bert-base-uncased")
        encoder_config.encoder_width = vision_width                         # new attributes
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token                       # 32
        Qformer = BertLMHeadModel(encoder_config)                           # decoder-only
        checkpoint = torch.load("../models/bert-base-uncased/model.pth", map_location=lambda storage, loc: storage)

        Qformer.load_state_dict(checkpoint['model_state_dict'], strict=True)
        query_tokens = nn.Parameter(                                        # query tokens initialize
            torch.zeros(1, num_query_token, encoder_config.hidden_size)     # [1, length, 768]
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)   # normal distribution
        return Qformer, query_tokens

    def forward(self, samples):                                 # num of samples ==  batch_size, every batch is a node
        behavior_embeds = torch.unsqueeze(samples[1], dim=1)    # samples[1]: [batch_size, 768], node features
        behavior_neighbors = samples[4]                         # [batch_size, 10]
        behavior_neighbors_length = samples[5].detach().tolist()            # [batch_size]
        behavior_neighbors_embeds = torch.tensor(self.GNN_embeddings[behavior_neighbors]).to(self.device) # [batch_size, 10, 768]
        text = samples[2]                                       # list with batch_size summaries

        behavior_embeds = behavior_embeds.to(self.device)       # [batch_size, 1, 768]
        behavior_neighbors_embeds = behavior_neighbors_embeds.to(self.device)   # [batch_size, 10, 768]
        behavior_atts = torch.ones(behavior_embeds.size()[:-1], dtype=torch.long).to(behavior_embeds.device)
                                                                # [batch_size, 1]
        behavior_neighbors_atts = [[1] * x + [0] * (behavior_neighbors.shape[-1] - x) \
            for x in behavior_neighbors_length]
        behavior_neighbors_atts = torch.tensor(np.array(behavior_neighbors_atts), dtype=torch.long).to(behavior_neighbors_embeds.device)

        query_tokens = self.query_tokens.expand(behavior_embeds.shape[0], -1, -1)   # repeated in each batch
                                                                # [batch_size, 32, 768]
        if self.use_neighbors:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,                          # [batch_size, 32, 768]
                # encoder_hidden_states=behavior_embeds,              # [batch_size, 1, 768], 执行cross-attention而不是self-attention
                encoder_hidden_states=behavior_neighbors_embeds,    # [batch_size, 10, 768], 执行cross-attention而不是self-attention
                # encoder_attention_mask=behavior_atts,               # [batch_size, 1]
                encoder_attention_mask=behavior_neighbors_atts,     # [batch_size, 10]
                use_cache=True,
                return_dict=True,
            )       # last_hidden_state: [batch_size, 32, 768]
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,                          # [batch_size, 32, 768]
                encoder_hidden_states=behavior_embeds,              # [batch_size, 1, 768], 执行cross-attention而不是self-attention
                # encoder_hidden_states=behavior_neighbors_embeds,    # [batch_size, 10, 768], 执行cross-attention而不是self-attention
                encoder_attention_mask=behavior_atts,               # [batch_size, 1]
                # encoder_attention_mask=behavior_neighbors_atts,     # [batch_size, 10]
                use_cache=True,
                return_dict=True,
            )       # last_hidden_state: [batch_size, 32, 768]
        # layer norm?
        behavior_feats = F.normalize(                           # [batch_size, 32, 256]
            self.behavior_proj(query_output.last_hidden_state), dim=-1
        )                                                       # (Image features)
        # tokenize summaries
        text_tokens = self.tokenizer(           # [batch_size, max_length]
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,        # 512 too short?
            return_tensors="pt",
        ).to(behavior_embeds.device)
        # encoding generated text summaries
        text_output = self.Qformer.bert(        # [batch_size, 512, 768]
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(                # [batch_size, 256]
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )                                       # fetch [CLS] feature for each summaries

        ###============== Image-text Contrastive ===================### (Image --> Query)
        behavior_feats_all = concat_all_gather(                 # if distributed
            behavior_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim] # torch.Size([8, 32, 256])
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(                       # [batch_size, 1, 32, 256] * [batch_size, 256, 1]
            behavior_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()                                   # [batch_size, batch_size, 32]
        # [batch_size, batch_size*num_gpu, num_query_tokens] 每组node之间有32个相似度

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)                # select from text dim 32个相似度中取最大的
        sim_i2t = sim_i2t / self.temp               # [batch_size, batch_size]

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(                     # [batch_size, 1, 1, 256] * [batch_size, 256, 32]
            text_feat.unsqueeze(1).unsqueeze(1), behavior_feats_all.permute(0, 2, 1)
        ).squeeze()                                 # [batch_size, batch_size, 32]

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)   # select from image dim
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = 0
        bs = behavior_embeds.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            behavior_embeds.device
        )                               # [batch_size], [0, 1, 2,.., batch_size - 1]

        loss_itc = (                   # contrast loss between summaries&query tokens
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching (classification task) ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)             # [batch_size, max_length]
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        behavior_embeds_world = all_gather_with_grad(behavior_embeds)               # graph feature [batch_size, 1, 768]
        behavior_neighbors_embeds_world = all_gather_with_grad(behavior_neighbors_embeds)   # [batch_size, 10, 768]
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)            # only for ddp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        behavior_embeds_neg = []
        behavior_neighbors_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()                   # randomly sample as negative samples
            behavior_embeds_neg.append(behavior_embeds_world[neg_idx])
            behavior_neighbors_embeds_neg.append(behavior_neighbors_embeds_world[neg_idx])
        behavior_embeds_neg = torch.stack(behavior_embeds_neg, dim=0)               # [batch_size, 1, 768]
        behavior_neighbors_embeds_neg = torch.stack(behavior_neighbors_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)                             # [batch_size, 512]
        text_atts_neg = torch.stack(text_atts_neg, dim=0)                           # [batch_size, 512]

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0     # [3 * batch_size, 512]
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],    # [3*batch_size, 512]
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)      # reapeat [3*batch_size, 32, 768]
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to( # [3*batch_size, 32]
            behavior_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)          # [3*batch_size, 512+32]

        behavior_embeds_all = torch.cat(                                                # [3*batch_size, 1, 768]
            [behavior_embeds, behavior_embeds_neg, behavior_embeds], dim=0
        )  # pos, neg, pos
        behavior_neighbors_embeds_all = torch.cat(                                      # [3*batch_size, 10 ,768]
            [behavior_neighbors_embeds, behavior_neighbors_embeds_neg, behavior_neighbors_embeds], dim=0
        )

        behavior_atts_all = torch.ones(behavior_embeds_all.size()[:-1], dtype=torch.long).to(
            behavior_embeds.device                                                      # [3*batch_size, 1]
        )
        behavior_neighbors_atts_all = [[1] * x + [0] * (behavior_neighbors.shape[-1] - x) \
            for x in behavior_neighbors_length] * 3
        behavior_neighbors_atts_all = torch.tensor(np.array(behavior_neighbors_atts_all), dtype=torch.long).to(behavior_neighbors_embeds.device)

        if self.use_neighbors:
            output_itm = self.Qformer.bert(                     # Q: [query tokens, text]; K,V: gnn feature
                text_ids_all,                                   # pos,pos,neg  [3*batch_size, 512]
                query_embeds=query_tokens_itm,                  # repeat in batch  [3*batch_size, 32, 768]  Q
                attention_mask=attention_mask_all,              # [3*batch_size, 512+32]
                # encoder_hidden_states=behavior_embeds_all,      # pos,neg,pos [3*batch_size, 1, 768]
                encoder_hidden_states=behavior_neighbors_embeds_all,
                # encoder_attention_mask=behavior_atts_all,
                encoder_attention_mask=behavior_neighbors_atts_all,
                return_dict=True,
            )   # last_hidden_state:    [3*batch_size, 32+512, 768]
        else:
            output_itm = self.Qformer.bert(                     # Q: [query tokens, text]; K,V: gnn feature
                text_ids_all,                                   # pos,pos,neg  [3*batch_size, 512]
                query_embeds=query_tokens_itm,                  # repeat in batch  [3*batch_size, 32, 768]  Q
                attention_mask=attention_mask_all,              # [3*batch_size, 512+32]
                encoder_hidden_states=behavior_embeds_all,      # pos,neg,pos [3*batch_size, 1, 768]
                # encoder_hidden_states=behavior_neighbors_embeds_all,
                encoder_attention_mask=behavior_atts_all,
                # encoder_attention_mask=behavior_neighbors_atts_all,
                return_dict=True,
            )   # last_hidden_state:    [3*batch_size, 32+512, 768]

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]  # [3*batch_size, 32, 768]
        vl_output = self.itm_head(vl_embeddings)        # [3*batch_size, 32, 2]
        logits = vl_output.mean(dim=1)                  # [3*batch_size, 2], pooling of 32 features

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(behavior_embeds.device)                    # [3 * batch_size] [[1, 1, .. 0, 0,.. 0, 0]]
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()       # [batch_size, 512] summary text
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id   # [DEC]
        labels = decoder_input_ids.masked_fill(                 # [batch_size, 512]
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            behavior_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return TranslatorOutput(                        # 3 optimized object
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=512,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        behavior_embeds = torch.unsqueeze(samples[1], dim=1).to(self.device)

        if not use_nucleus_sampling:
            behavior_embeds = behavior_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        behavior_atts = torch.ones(behavior_embeds.size()[:-1], dtype=torch.long).to(
            behavior_embeds.device
        )

        model_kwargs = {
            "encoder_hidden_states": behavior_embeds,
            "encoder_attention_mask": behavior_atts,
        }

        input_ids = (
            torch.LongTensor(samples[1].size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(behavior_embeds.device)
        )
        query_tokens = self.query_tokens.expand(behavior_embeds.shape[0], -1, -1).to(behavior_embeds.device)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):

        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return TranslatorOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):                          # initialize
        # Behavior
        behavior_length = cfg.get("behavior_length", 384)

        # Text
        max_txt_len = cfg.get("max_txt_len", 32)

        # Q-Former
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        model = cls(                                    # define and initialize Qformer
            num_features=behavior_length,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)          # load BERT checkpoint

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    def load_from_pretrained(self, url_or_filename):
        if url_or_filename:
            checkpoint = torch.load(url_or_filename, map_location=lambda storage, loc: storage)
            if "model_state_dict" in checkpoint.keys():
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            msg = self.load_state_dict(state_dict, strict=False)

            logging.info("load checkpoint from %s" % url_or_filename)

            return msg

        return
