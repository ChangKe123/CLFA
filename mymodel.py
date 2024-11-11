# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,Linear
import torch.nn.functional as F
from transformers import BertModel
from vit.modeling import VisionTransformer
from vit.modeling import CONFIGS
import numpy as np

class CoAttention(nn.Module):
	def __init__(self,text_hidden_size, k):
		super(CoAttention, self).__init__()
		self.w_b=Linear(text_hidden_size, text_hidden_size, bias=False)
		self.w_v=Linear(text_hidden_size, k, bias=False)
		self.w_q=Linear(text_hidden_size, k,bias=False)
		self.w_hv=Linear(k,1, bias=False)
		self.w_hq=Linear(k,1, bias=False)
		self.dropout = nn.Dropout(0.1)
		
		self.w_s=Linear(text_hidden_size,text_hidden_size,bias=False)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, text_hidden_states, image_hidden_states, text_attention_mask):
		# text_hidden_states  b*10*768
		# image_hidden_states b*16*768
		
		affinity_matrix=torch.einsum('bxe,bye->bxy',[self.w_b(text_hidden_states), image_hidden_states])
		wv_v=self.w_v(image_hidden_states)
		wq_q=self.w_q(text_hidden_states)
		
		wqqc=torch.einsum('bxk,bxy->byk', [wq_q, affinity_matrix])
		wvvc=torch.einsum('byk,bxy->bxk', [wv_v, affinity_matrix])
		
		h_v=torch.tanh(wv_v+wqqc)
		h_q=torch.tanh(wq_q+wvvc)

		attention_v=torch.softmax(self.w_hv(h_v).squeeze(2), dim=-1)
		attention_q=torch.softmax(self.w_hq(h_q).squeeze(2), dim=-1)
		
		context_v=torch.einsum('bx,bxd->bd', [attention_v, image_hidden_states])
		context_q=torch.einsum('by,byd->bd', [attention_q, text_hidden_states])

		feature=torch.tanh(self.w_s(context_v+context_q))
		# context_layer b*75*768
		return feature

class VitBERT(nn.Module):
	def __init__(self, args):
		super(VitBERT, self).__init__()
		self.bert = BertModel.from_pretrained(args.bert_pretrained_dir)
		config = CONFIGS[args.model_type]
		
		self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=10)
		self.model.load_from(np.load(args.vit_pretrained_dir))
		
		self.coattention=CoAttention(768, 512)
		self.classifier = nn.Linear(768, 2)

	def forward(self, input_ids, vision_features, input_mask, added_attention_mask, hashtag_input_ids,
				hashtag_input_mask, labels=None):
		sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
		img_features,_=self.model.transformer(vision_features)
		
		co_attn_output=self.coattention(sequence_output, img_features, input_mask)
		
		#pooled_output = self.dropout(co_attn_output)
		logits = self.classifier(co_attn_output)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, 2), labels.view(-1))
			return loss
		else:
			return logits


class BertOnly(nn.Module):
	def __init__(self, args):
		super(BertOnly, self).__init__()
		self.bert = BertModel.from_pretrained(args.bert_pretrained_dir)
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(768, 2)

	def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
				hashtag_input_mask, labels=None):
		sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, 2), labels.view(-1))
			return loss
		else:
			return logits


class VitOnly(nn.Module):
	def __init__(self, args):
		super(VitOnly, self).__init__()
		config = CONFIGS[args.model_type]
		self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=10)
		self.model.load_from(np.load(args.vit_pretrained_dir))
		self.vismap2text = nn.Linear(768, 768)
		self.classifier = nn.Linear(768, 2)

	def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
				hashtag_input_mask, labels=None):
		# b*49*2048
		img_features,_=self.model.transformer(visual_embeds_att)
		vis_embed_map = self.vismap2text(img_features)
		vis_embed_map = vis_embed_map.mean(1)
		logits = self.classifier(vis_embed_map)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, 2), labels.view(-1))
			return loss
		else:
			return logits
