import os
ocr_dir='../kedaxunfei_ocr.txt'
captions_dir='../image_captions.txt'
data_dir='datasets/sarcasm/'
lmt=2000

hh={}
hh2={}
with open(ocr_dir, 'r') as f:
	for line in f.readlines():
		a=eval(line)
		hh[a[0]]=a[1].lower()
	f.close()

with open(captions_dir, 'r') as f:
	for line in f.readlines():
		a=eval(line)
		hh2[a[0]]=a[1].lower()
	f.close()

def load_data(data_path):
	w=open(data_path+'.ocr', 'w')
	with open(data_path ,'r') as f:
		for line in f.readlines():
			a=eval(line)
			text=''
			if a[0] in hh:
				text+=hh[a[0]]

			if a[0] in hh2:
				text+=' [SEP] '+hh2[a[0]]
			
			out=[a[0], a[1], text, a[-1]]
			w.write(str(out)+'\n')
		f.close()
		w.close()


load_data(data_dir+'/train.txt')
load_data(data_dir+'/valid.txt')
load_data(data_dir+'/test.txt')

