import numpy as np
import spacy
import pickle
from tqdm import tqdm
import math
from copy import deepcopy
from collections import defaultdict
from transformers import BertTokenizerFast
from nltk.corpus import wordnet as wn
from torchvision import transforms

tokenizer = BertTokenizerFast.from_pretrained('/home/zhangming/project/models/bert-base-uncased')
nlp = spacy.load('en_core_web_sm')
y = math.exp(1)

def dependency_adj_matrix_2(doc):
	mat = defaultdict(list,[])
	for t in doc:
		for child in t.children:
			mat[child.i].append(t.i)
			mat[t.i].append(child.i)
	return mat

def load_sentic_word():
	"""
	load senticNet
	"""
	path = './senticnet.xls'
	senticNet = {}
	senticNet2={}
	fp = open(path, 'r')
	cnt=0
	for line in fp:
		cnt+=1
		if cnt==1:
			continue
		line = line.strip()
		if not line:
			continue
		arr = line.split('\t')
		word, sentic = arr[0], arr[8]
		senticNet[word] = float(sentic)
		words=word.split('_')
		if len(words)>=2:
			prefix=words[0]+'_'+words[1]
			if prefix in senticNet2:
				senticNet2[prefix].append((words, float(sentic)))
			else:
				senticNet2[prefix]=[(words, float(sentic))]

	fp.close()
	return senticNet, senticNet2

senticNet, senticNet2 = load_sentic_word()

def get_sentic_score(si, sj):
	return abs(float(si - sj)) * y**(-1*si*sj)

def bert_tokens_to_pos(tokens):
	dc=[]
	for i, t in enumerate(tokens):
		tk=t.replace('##','')
		if t=='[UNK]':
			dc.append(i)
			continue
		for j in range(len(tk)):
			dc.append(i)
	return dc
	
def bert_tokens_to_pos_span(tokens, l):
	dc=[0]*l
	last=0
	for i, t in enumerate(tokens):
		for j in range(last, t[0]):
			dc[j]=i
		for j in range(t[0], t[1]):
			dc[j]=i
		last=t[1]

	for j in range(last, l):
		dc[j]=i

	return dc

def bert_pos_to_tokens(tokens):
	dc=[0]*len(tokens)
	for i, t in enumerate(tokens):
		dc[i]=t[0]
	return dc

def spacy_tokens_to_pos(doc, l):
	dc=[0]*l
	for i, t in enumerate(doc):
		dc[i]=t.idx
	return dc



def word_to_pos(text):
	words=text.split(' ')
	filt_words=[]
	pos=[]
	c=0
	for idx, w in enumerate(words):
		for j in range(len(w)):
			pos.append(c)
		if idx!=0:
			pos.append(c)
		if len(w.strip())!=0:
			c+=1
			filt_words.append(w)
	
	assert len(pos)==len(text)

	return pos, filt_words


def get_word_sentics(words):
	sentic_scores=[-1]*len(words)
	idx=0
	while idx < len(words):
		w=words[idx]
		maxlen=0
		score=-1
		if idx+1<len(words):
			w2=w+'_'+words[idx+1]

			if w2 in senticNet2:
				for sentic_words, sentic_score in senticNet2[w2]:
					flg=True
					for j in range(len(sentic_words)):
						if idx+j>=len(words):
							flg=False
							break
						if words[idx+j]!=sentic_words[j]:
							flg=False
							break
					if flg:
						if len(sentic_words)>maxlen:
							maxlen=len(sentic_words)
							score=sentic_score
		
		if maxlen==0:
			if w in senticNet:
				score=senticNet[w]
				sentic_scores[idx]=score
			idx+=1
		else:
			for j in range(maxlen):
				sentic_scores[idx+j]=score
			idx+=maxlen
	
	return sentic_scores

def get_antonyms(synset):
	for lemma in synset.lemmas():
		if lemma.antonyms():
			return lemma.antonyms()[0]
	return None

def generate_cross_graph(text1, text2):
	
	t1_token_result = tokenizer.encode_plus(text1,return_offsets_mapping=True, add_special_tokens=False)
	t2_token_result = tokenizer.encode_plus(text2,return_offsets_mapping=True, add_special_tokens=False)
	t1_token=t1_token_result['input_ids']
	t2_token=t2_token_result['input_ids']
	t1_token_span=t1_token_result['offset_mapping']
	t2_token_span=t2_token_result['offset_mapping']
	wordnet_graph = np.zeros((len(t1_token), len(t2_token))).astype('float32')
	senticnet_graph = np.zeros((len(t1_token), len(t2_token))).astype('float32')
	t1_pos_dict=bert_pos_to_tokens(t1_token_span)
	t2_pos_dict=bert_pos_to_tokens(t2_token_span)
	
	t1_words_pos, t1_filt_words=word_to_pos(text1)
	t2_words_pos, t2_filt_words=word_to_pos(text2)
	
	t1_words_sentic =get_word_sentics(t1_filt_words)
	t2_words_sentic =get_word_sentics(t2_filt_words)

	for i in range(len(t1_token)):
		for j in range(len(t2_token)):
			w1=t1_filt_words[t1_words_pos[t1_pos_dict[i]]]		
			w2=t2_filt_words[t2_words_pos[t2_pos_dict[j]]]

			sentic1=t1_words_sentic[t1_words_pos[t1_pos_dict[i]]]
			sentic2=t2_words_sentic[t2_words_pos[t2_pos_dict[j]]]
			
			s1_syn=wn.synsets(w1)
			s2_syn=wn.synsets(w2)
			
			if sentic1!=-1 and  sentic2!=-1 and w1!=w2:
				senticnet_graph[i][j]=get_sentic_score(sentic1, sentic2)
			
			if len(s1_syn)!=0 and len(s2_syn)!=0:
				antonyms1=get_antonyms(s1_syn[0])
				antonyms2=get_antonyms(s2_syn[0])
				if antonyms1 and antonyms2:
					wordnet_graph[i][j]=(wn.path_similarity(s1_syn[0], antonyms2.synset()) + wn.path_similarity(s2_syn[0], antonyms1.synset()))/2

	
	return senticnet_graph, wordnet_graph, t2_token



def generate_graph(line):
	line = line.lower().strip()
	bert_token_result = tokenizer.encode_plus(line,return_offsets_mapping=True, add_special_tokens=False)
	bert_token=bert_token_result['input_ids']
	bert_token_span=bert_token_result['offset_mapping']
	document = nlp(line)
	spacy_token = [str(x) for x in document]
	spacy_len = len(spacy_token)
	bert_len = len(bert_token)
	outter_graph = np.zeros((bert_len, bert_len)).astype('float32')
	split_link = []
	ii = 0
	jj = 0
	pre = []
	s = ""
	bert_pos_dict=bert_tokens_to_pos_span(bert_token_span, len(line))
	spacy_pos_dict=spacy_tokens_to_pos(document, len(line))

	
	for i, spacy_ in enumerate(spacy_token):
		pf=bert_pos_dict[spacy_pos_dict[i]]
		if i+1 ==len(spacy_token):
			pt=len(bert_token)
		else:
			pt=bert_pos_dict[spacy_pos_dict[i+1]]
		split_link.append(list(range(pf, pt)))
	
	flag = False
	mat = dependency_adj_matrix_2(document)
	for key,linked in mat.items():
		for x in split_link[int(key)]:
			for link in linked:
				for y in split_link[int(link)]:
					outter_graph[x][y] = 1	 
	
	tokens = bert_token
	inner_graph = np.identity(bert_len).astype('float32')
	for link in split_link:
		for x in link:
			for y in link:
				inner_graph[x][y] = 1

	
	graph1 = inner_graph + outter_graph
	for i in range(len(graph1)):
		for j in range(len(graph1)):
			if graph1[i][j] > 0:
				graph1[i][j] = 1
	
	return graph1,tokens


device='cuda'
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("/home/zhangming/project/models/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("/home/zhangming/project/models/clip-vit-base-patch32")
model.eval()
model.to(device)
import os


def process_clip(img, text):
	if not os.path.exists(img):
		return [], []
	image = Image.open(img)
	inputs=proc(text=[text], images=image, return_tensors='pt', truncation=True, max_length=77)
	inputs.to(device)
	outputs=model(**inputs)
	tf=outputs.text_embeds[0].detach().cpu().numpy().tolist()
	vf=outputs.image_embeds[0].detach().cpu().numpy().tolist()

	return tf, vf
	


def process(filename,outfile = ""):
	with open("./datasets/sarcasm/{}.txt".format(filename),'r',encoding='utf-8') as fin:
		lines = fin.readlines()
		cnt = 0
		lines = [x.strip() for x in lines]
		dic = dict()
		for i in tqdm(range(len(lines))):
			cnt+=1
			line = lines[i]
			data = eval(line)
			if 'train' in filename:
				id_,text,knowledge,label = data
			else:
				id_,text,knowledge,label = data
				
			sentic_graph, wordnet_graph, knowledge_tokens = generate_cross_graph(text.lower(),knowledge)
			self_sentic_graph, self_wordnet_graph, tokens = generate_cross_graph(text.lower(),text.lower())
			
			image='../dataset_image/'+str(id_)+'.jpg'
			text_embed, image_embed=process_clip(image, text.lower().strip())
			
			dic[id_] = {'image_id': id_, 'text':text,'label':int(label), 'knowledge': knowledge, 'tokens':tokens, 'knowledge_tokens': knowledge_tokens,'text_graph':self_sentic_graph, 'sentic_graph':sentic_graph, 'wordnet_graph':wordnet_graph, 'text_embed':text_embed, 'image_embed':image_embed}
		pickle.dump(dic,open("./datasets/sarcasm/{}{}".format(filename,outfile),'wb'))

process("train",".with_graph.pkl")
process("test",".with_graph.pkl")
process("valid",".with_graph.pkl")
