import logging
import os
import pickle
import torch
import wordninja
from PIL import Image
from torchvision import transforms
import numpy as np
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
	def __init__(self, text, img_id, knowledge, text_tokens, knowledge_tokens, label=None, self_graph=None, cross_graph=None, text_embeds=None, image_embeds=None):
		"""Constructs an InputExample."""
		self.text = text
		self.knowledge = knowledge
		self.text_tokens=text_tokens
		self.knowledge_tokens=knowledge_tokens
		self.self_graph=self_graph
		self.cross_graph=cross_graph
		self.img_id = img_id
		self.label = label
		self.text_embeds=text_embeds
		self.image_embeds=image_embeds


class MMInputFeatures(object):
	def __init__(self, input_ids,
				 input_mask,
				 added_input_mask,
				 img_feat,
				 hashtag_input_ids,
				 hashtag_input_mask,
				 label_id,
				 self_graph,
				 cross_graph,
				 text_embeds,
				 image_embeds):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.added_input_mask = added_input_mask
		self.img_feat = img_feat
		self.hashtag_input_ids = hashtag_input_ids
		self.hashtag_input_mask = hashtag_input_mask
		self.label_id = label_id
		self.self_graph=self_graph
		self.cross_graph=cross_graph
		self.text_embeds=text_embeds
		self.image_embeds=image_embeds

class Processer():
	def __init__(self, data_dir, image_path, model_select, max_seq_length, max_hashtag_length, cross_attn_mode, lambda1, lambda2, gamma):
		self.model_select = model_select
		self.max_seq_length = max_seq_length
		self.max_hashtag_length = max_hashtag_length
		self.image_path = image_path
		self.data_dir = data_dir
		self.cross_attn_mode = cross_attn_mode
		self.lambda1=lambda1
		self.lambda2=lambda2
		self.gamma=gamma

	def get_train_examples(self):
		return self._create_examples(os.path.join(self.data_dir, "train.with_graph.pkl"))

	def get_eval_examples(self):
		return self._create_examples(os.path.join(self.data_dir, "valid.with_graph.pkl"))

	def get_test_examples(self):
		return self._create_examples(os.path.join(self.data_dir, "test.with_graph.pkl"))

	def get_labels(self):
		return [0, 1]

	def _create_examples(self, data_file):
		"""Creates examples for the training and dev sets."""
		examples = []
		data = pickle.load(open(data_file,'rb'))
		for key, value in data.items():
			tmpLS = value['text'].split()
			if "sarcasm" in tmpLS:
				continue
			if "sarcastic" in tmpLS:
				continue
			if "reposting" in tmpLS:
				continue
			if "<url>" in tmpLS:
				continue
			if "joke" in tmpLS:
				continue
			if "humour" in tmpLS:
				continue
			if "humor" in tmpLS:
				continue
			if "jokes" in tmpLS:
				continue
			if "irony" in tmpLS:
				continue
			if "ironic" in tmpLS:
				continue
			if "exgag" in tmpLS:
				continue
			img_id = value['image_id']
			text = value['text']
			label =value['label']
			knowledge = value['knowledge']
			text_tokens= value['tokens']
			knowledge_tokens = value['knowledge_tokens']
			sentic_graph=value['sentic_graph']
			text_graph=value['text_graph']*self.gamma
			wordnet_graph =value['wordnet_graph']
			text_embeds=value['text_embed']
			image_embeds=value['image_embed']
			if self.cross_attn_mode =='sentic':
				cross_graph=sentic_graph*self.lambda1
			elif self.cross_attn_mode =='wordnet':
				cross_graph=wordnet_graph*self.lambda2
			else:
				cross_graph=sentic_graph*self.lambda1 + wordnet_graph*self.lambda2
			examples.append(InputExample(text=text, img_id=img_id, knowledge=knowledge, label=label, text_tokens=text_tokens, knowledge_tokens=knowledge_tokens, self_graph=text_graph, cross_graph=cross_graph, text_embeds=text_embeds, image_embeds=image_embeds))
		return examples

	def image_process(self, image_path, transform):
		image = Image.open(image_path).convert('RGB')
		image = transform(image)
		return image

	def get_image_text(self):
		image_text = {}
		with open(self.image_path) as f:
			for line in f.readlines():
				sp = line.strip().split()
				if sp[0] not in image_text.keys():
					image_text[sp[0]] = " ".join(sp)
		return image_text

	def convert_mm_examples_to_features(self, examples, label_list, tokenizer):
		label_map = {label: i for i, label in enumerate(label_list)}
		features = []

		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406),
								 (0.229, 0.224, 0.225))])

		for (ex_index, example) in enumerate(examples):
			'''
			hashtags = []
			tokens = []

			sent = example.text.split()
			i = 0
			while i < len(sent):
				if sent[i] == "#" and i < len(sent) - 1:
					while sent[i] == "#" and i < len(sent) - 1:
						i += 1
					if sent[i] != "#":
						temp = wordninja.split(sent[i])
						for _ in temp:
							hashtags.append(_)
				else:
					if sent[i] != "#":
						temp = wordninja.split(sent[i])
						for _ in temp:
							tokens.append(_)
					i += 1
			tokens = " ".join(tokens)
			hashtags = " ".join(hashtags) if len(hashtags) != 0 else "None"
			tokens = tokenizer.tokenize(tokens)
			hashtags = tokenizer.tokenize(example.ocr_text)
			'''

			#####
			# image_text = None
			# image_text_dic = get_image_text()

			# if example.img_id in image_text_dic:
			#	 image_text = list(image_text_dic[example.img_id])
			# else:
			#	 image_text = ["None"]
			#####
			
			tokens=example.text_tokens
			hashtags=example.knowledge_tokens
			self_graph=example.self_graph
			cross_graph=example.cross_graph
			text_embeds=example.text_embeds
			image_embeds=example.image_embeds
			if len(tokens) > self.max_seq_length - 2:
				tokens = tokens[:(self.max_seq_length - 2)]
				self_graph=self_graph[:(self.max_seq_length - 2), :(self.max_seq_length - 2)]
				cross_graph=cross_graph[:(self.max_seq_length - 2),:]
			if len(hashtags) > self.max_hashtag_length - 2:
				hashtags = hashtags[:(self.max_hashtag_length - 2)]
				cross_graph=cross_graph[:,:(self.max_hashtag_length - 2)]
			if len(tokens) < self.max_seq_length - 2:
				self_graph=np.pad(self_graph, [0, self.max_seq_length - 2-len(tokens)])
				cross_graph=np.pad(cross_graph, [[0, self.max_seq_length - 2-len(tokens)], [0,0]])

			if len(hashtags) < self.max_hashtag_length - 2:
				cross_graph=np.pad(cross_graph, [[0,0], [0, self.max_hashtag_length - 2-len(hashtags)]])

			self_graph=np.pad(self_graph, [[1,1], [1,1]])
			self_graph[0,0]=1
			self_graph[-1,-1]=1
			cross_graph=np.pad(cross_graph, [[1,1], [1,1]])

			input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
			input_mask = [1] * len(input_ids)
			added_input_mask = [1] * (len(input_ids) + 50)
			padding = [0] * (self.max_seq_length - len(input_ids))
			input_ids += padding
			input_mask += padding
			added_input_mask += padding

			hashtag_input_ids = [tokenizer.cls_token_id] + hashtags + [tokenizer.sep_token_id]
			hashtag_input_mask = [1] * len(hashtag_input_ids)
			hashtag_padding = [0] * (self.max_hashtag_length - len(hashtag_input_ids))
			hashtag_input_ids += hashtag_padding
			hashtag_input_mask += hashtag_padding
			assert len(input_ids) == self.max_seq_length
			assert len(input_mask) == self.max_seq_length

			assert len(hashtag_input_ids) == self.max_hashtag_length
			assert len(hashtag_input_mask) == self.max_hashtag_length
			label_id = label_map[example.label]

			# process images
			image_name = example.img_id
			image_path = os.path.join(self.image_path, image_name + ".jpg")
			image = self.image_process(image_path, transform)  # 3*224*224

			features.append(MMInputFeatures(input_ids=input_ids,
											input_mask=input_mask,
											added_input_mask=added_input_mask,
											img_feat=image,
											hashtag_input_ids=hashtag_input_ids,
											hashtag_input_mask=hashtag_input_mask,
											label_id=label_id, self_graph=self_graph, cross_graph=cross_graph, text_embeds=text_embeds, image_embeds=image_embeds))
			if ex_index % 1000 == 0:
				logger.info("processed image num: " + str(ex_index) + " **********")
		all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
		all_added_input_mask = torch.tensor([f.added_input_mask for f in features], dtype=torch.long)
		all_img_feats = torch.stack([f.img_feat for f in features])
		all_hashtag_input_ids = torch.tensor([f.hashtag_input_ids for f in features], dtype=torch.long)
		all_hashtag_input_mask = torch.tensor([f.hashtag_input_mask for f in features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
		all_extra_self_attention = torch.tensor([f.self_graph for f in features], dtype=torch.float)
		all_extra_cross_attention = torch.tensor([f.cross_graph for f in features], dtype=torch.float)
		all_text_embeds = torch.tensor([f.text_embeds for f in features], dtype=torch.float)
		all_image_embeds = torch.tensor([f.image_embeds for f in features], dtype=torch.float)
		return all_input_ids, all_input_mask, all_added_input_mask, all_img_feats, all_hashtag_input_ids, all_hashtag_input_mask, all_label_ids, all_extra_self_attention, all_extra_cross_attention, all_text_embeds, all_image_embeds
