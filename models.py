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
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertModel
from vit.modeling import VisionTransformer
from vit.modeling import CONFIGS
import numpy as np
from mlp import MLP

def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertPooler(nn.Module):
	def __init__(self):
		super(BertPooler, self).__init__()
		self.dense = nn.Linear(512, 512)
		self.activation = nn.Tanh()

	def forward(self, hidden_states):
		first_token_tensor = hidden_states[:, 0]
		pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output


class BertLayerNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		"""
		super(BertLayerNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.bias = nn.Parameter(torch.zeros(hidden_size))
		self.variance_epsilon = eps

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.weight * x + self.bias


class BertIntermediate(nn.Module):
	def __init__(self):
		super(BertIntermediate, self).__init__()
		self.dense = nn.Linear(512, 2048)

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = gelu(hidden_states)
		return hidden_states


class BertCoAttention(nn.Module):
	def __init__(self):
		super(BertCoAttention, self).__init__()
		self.num_attention_heads = 8
		self.hidden_size = 512
		self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.query = nn.Linear(self.hidden_size, self.all_head_size)
		self.key = nn.Linear(self.hidden_size, self.all_head_size)
		self.value = nn.Linear(self.hidden_size, self.all_head_size)
		self.dropout = nn.Dropout(0.1)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, extra_attention=None):
		# s2_attention_mask  b*1*1*49
		mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
		mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
		mixed_value_layer = self.value(s2_hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
		key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
		value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # b*12*75*49
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + s2_attention_mask

		if not extra_attention is None:
			attention_scores = attention_scores + extra_attention
		# atention_scores b*12*75*49
		# Normalize the attention scores to probabilities.
		# b*12*75*49
		attention_probs = nn.Softmax(dim=-1)(attention_scores)
		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		# context_layer b*12*75*64
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		# context_layer b*75*768
		return context_layer


class BertOutput(nn.Module):
	def __init__(self, res):
		super(BertOutput, self).__init__()
		self.dense = nn.Linear(2048, 512)
		self.LayerNorm = BertLayerNorm(512, eps=1e-12)
		self.dropout = nn.Dropout(0.1)
		self.res = res

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		if self.res:
			hidden_states = self.LayerNorm(hidden_states + input_tensor)
		else:
			hidden_states = self.LayerNorm(hidden_states)

		return hidden_states


class BertSelfOutput(nn.Module):
	def __init__(self, res):
		super(BertSelfOutput, self).__init__()
		self.dense = nn.Linear(512, 512)
		self.LayerNorm = BertLayerNorm(512, eps=1e-12)
		self.dropout = nn.Dropout(0.1)
		self.res = res

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		if self.res:
			hidden_states = self.LayerNorm(hidden_states + input_tensor)
		else:
			hidden_states = self.LayerNorm(hidden_states)

		return hidden_states


class BertCrossAttention(nn.Module):
	def __init__(self, res):
		super(BertCrossAttention, self).__init__()
		self.bertCoAttn = BertCoAttention()
		self.output = BertSelfOutput(res)

	def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask, extra_attention=None):
		s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask, extra_attention)
		attention_output = self.output(s1_cross_output, s1_input_tensor)
		return attention_output


class BertCrossAttentionLayer(nn.Module):
	def __init__(self, res):
		super(BertCrossAttentionLayer, self).__init__()
		self.textSelfAttn = BertCrossAttention(res)
		self.imageCrossAttn = BertCrossAttention(res)
		self.knowledgeCrossAttn = BertCrossAttention(res)
		self.intermediate = BertIntermediate()
		self.output = BertOutput(res)

	def forward(self,  text_hidden_states, image_hidden_states, knowledge_hidden_states,  text_mask, image_mask,  knowledge_mask, self_extra_attention=None, cross_extra_attention=None):
		text_self_attention_output = self.textSelfAttn(text_hidden_states, text_hidden_states , text_mask, self_extra_attention)
		imag_cross_attention_output = self.imageCrossAttn(text_self_attention_output, image_hidden_states , image_mask)
		knowledge_cross_attention_output = self.knowledgeCrossAttn(imag_cross_attention_output, knowledge_hidden_states, knowledge_mask, cross_extra_attention)
		# b*75*768
		intermediate_output = self.intermediate(knowledge_cross_attention_output)
		# b*75*3072
		layer_output = self.output(intermediate_output, knowledge_cross_attention_output)
		# b*75*3072
		return layer_output


class BertCrossEncoder(nn.Module):
	def __init__(self, layer_num=3, res=True):
		super(BertCrossEncoder, self).__init__()
		layer = BertCrossAttentionLayer(res)
		self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

	def forward(self, text_hidden_states, image_hidden_states, knowledge_hidden_states, text_mask, image_mask,  knowledge_mask, self_extra_attention=None, cross_extra_attention=None):
		for layer_module in self.layer:
			text_hidden_states = layer_module(text_hidden_states, image_hidden_states, knowledge_hidden_states, text_mask,  image_mask,  knowledge_mask, self_extra_attention=self_extra_attention, cross_extra_attention=cross_extra_attention)
		return text_hidden_states


class VitBERT(nn.Module):
	def __init__(self, args):
		super(VitBERT, self).__init__()
		self.bert = BertModel.from_pretrained(args.bert_pretrained_dir)
		self.tanh = nn.Tanh()

		self.alpha=args.alpha
		self.beta=args.beta
		self.temperature=args.temperature
		
		self.mlp_text=MLP(768, 512, 512*3)
		self.mlp_image=MLP(768, 512, 512*3)

		self.criterion=CrossEntropyLoss()
		self.device=args.device
		
		self.attention_net = BertCrossEncoder(layer_num=args.attn_layer_num)
		self.text_pooler = BertPooler()
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(512, 2)
		
		config = CONFIGS[args.model_type]
		self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=10)
		self.model.load_from(np.load(args.vit_pretrained_dir))

	def avg_pooling(self, features, mask):
		return torch.sum(features*(mask.unsqueeze(2)),dim=1)/torch.sum(mask, dim=1,keepdim=True)

	def max_pooling(self, features, mask):
		return torch.max(features*(mask.unsqueeze(2)),dim=1).values
	
	def contrastive(self, features, mask, clip_features):
		batch_size, _, __ = features.shape
		features=self.avg_pooling(features, mask)

		features=nn.functional.normalize(features, dim=1)
		clip_features=nn.functional.normalize(clip_features, dim=1)

		logits1=torch.einsum('nc,mc->nm', [features, clip_features])/self.temperature
		logits2=torch.einsum('nc,mc->nm', [clip_features, features])/self.temperature
		labels=torch.tensor(list(range(0, batch_size))).to(self.device)

		con_loss1=self.criterion(logits1, labels)
		con_loss2=self.criterion(logits2, labels)
		
		return 0.5*con_loss1+0.5*con_loss2

	def forward(self, input_ids, vision_features, input_mask, added_attention_mask, hashtag_input_ids, 
				hashtag_input_mask, labels=None,  self_extra_attention=None, cross_extra_attention=None, pooling_type=0, text_embeds=None, image_embeds=None, train_step_rate=1):
		sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, return_dict=False)
		hashtag_output, hashtag_pooled_output = self.bert(input_ids=hashtag_input_ids, token_type_ids=None,
																  attention_mask=hashtag_input_mask, return_dict=False)
		
		if not self_extra_attention is None:
			self_extra_attention=self_extra_attention.unsqueeze(1)
		if not cross_extra_attention is None:
			cross_extra_attention=cross_extra_attention.unsqueeze(1)

		extended_input_mask = input_mask.unsqueeze(1).unsqueeze(2)
		extended_input_mask = extended_input_mask.to(dtype=next(self.parameters()).dtype)
		extended_input_mask = (1.0 - extended_input_mask) * -10000.0

		# added_attention_mask batch_size*124
		img_mask = added_attention_mask[:, :50]
		extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
		extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)
		extended_img_mask = (1.0 - extended_img_mask) * -10000.0
		
		extended_ocr_mask = hashtag_input_mask.unsqueeze(1).unsqueeze(2)
		extended_ocr_mask = extended_ocr_mask.to(dtype=next(self.parameters()).dtype)
		extended_ocr_mask = (1.0 - extended_ocr_mask) * -10000.0
		# img_mask # b*49
		# img_mask # b*1*1*49
		# text作为query， image作为key和value
		
		img_features,_=self.model.transformer(vision_features)
		
		sequence_output = self.mlp_text(sequence_output)
		hashtag_output = self.mlp_text(hashtag_output)
		img_features = self.mlp_image(img_features)
		

		attn_output =self.attention_net(sequence_output, img_features, hashtag_output, extended_input_mask, extended_img_mask, extended_ocr_mask, self_extra_attention, cross_extra_attention)
		
		
		contrast_loss_img = self.contrastive(img_features, img_mask, image_embeds)
		contrast_loss_text = self.contrastive(sequence_output, input_mask, text_embeds)
		contrast_loss = contrast_loss_img + contrast_loss_text


		# batchsize*49*2048
		# b*49*768
		# b*75*768
		'''
		# b*75*12
		C = self.tanh(torch.matmul(torch.matmul(sequence_output, self.W_b), hashtag_output.transpose(1,2)))
		# C: b*12
		C, _ = torch.max(C, dim=1)
		attn = F.softmax(C, dim=-1)
		# b*1*768
		hashtag_text_cross_attn = torch.matmul(attn.unsqueeze(1), hashtag_output)
		# b*1*768
		'''
		if pooling_type==0:
			text_pooled_output = self.text_pooler(attn_output)
		elif pooling_type==1:
			text_pooled_output = self.avg_pooling(attn_output,input_mask)
		else:
			text_pooled_output = self.max_pooling(attn_output,input_mask)

		pooled_output = self.dropout(text_pooled_output)
		logits = self.classifier(pooled_output)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			sarcasm_loss = loss_fct(logits.view(-1, 2), labels.view(-1))

			loss=self.alpha*contrast_loss + self.beta*sarcasm_loss
			return loss
		else:
			return logits, pooled_output


class Res_BERT(nn.Module):
	def __init__(self):
		super(Res_BERT, self).__init__()
		self.bert = BertModel.from_pretrained('/home/zhangming/project/models/bert-base-cased')
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(768, 2)
		self.vismap2text = nn.Linear(2048, 768)

	def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
				hashtag_input_mask, labels=None):
		# b*75*768
		sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
		# batchsize*49*2048
		vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
		# b*49*768
		visual = self.vismap2text(vis_embed_map)
		# b*1*768
		res = torch.cat([sequence_output, visual], dim=1).mean(1)
		pooled_output = self.dropout(res)
		logits = self.classifier(pooled_output)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, 2), labels.view(-1))
			return loss
		else:
			return logits


class BertOnly(nn.Module):
	def __init__(self):
		super(BertOnly, self).__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
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


class ResNetOnly(nn.Module):
	def __init__(self):
		super(ResNetOnly, self).__init__()
		self.vismap2text = nn.Linear(2048, 768)
		self.classifier = nn.Linear(768, 2)

	def forward(self, input_ids, visual_embeds_att, input_mask, added_attention_mask, hashtag_input_ids,
				hashtag_input_mask, labels=None):
		vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
		# b*49*2048
		vis_embed_map = self.vismap2text(vis_embed_map)
		vis_embed_map = vis_embed_map.mean(1)
		logits = self.classifier(vis_embed_map)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, 2), labels.view(-1))
			return loss
		else:
			return logits
