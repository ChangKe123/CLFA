import os
ocr_dir='../image_captions.txt'
data_dir='datasets/sarcasm/'
lmt=2000

hh={}
with open(ocr_dir, 'r') as f:
	for line in f.readlines():
		a=eval(line)
		hh[a[0]]=a[1].lower()
	f.close()


def load_ocr(ocr_path):
	text=''
	if not os.path.exists(ocr_path):
		return text
	limit=512
	c=0
	with open(ocr_path, 'r') as f:
		for line in f.readlines():
			a=eval(line)
			c+=1
			if len(text)!=0:
				text+=' '
			text+=a[1]
			if c>=limit:
				break
	return text.lower()

def load_data(data_path):
	w=open(data_path+'.ocr', 'w')
	with open(data_path ,'r') as f:
		for line in f.readlines():
			a=eval(line)
			text=''
			if a[0] in hh:
				text=hh[a[0]]
			
			out=[a[0], a[1], text, a[-1]]
			if out[-1] != a[-1]:
				print(a)
				print(out)
			w.write(str(out)+'\n')
		f.close()
		w.close()


load_data(data_dir+'/train.txt')
load_data(data_dir+'/valid.txt')
load_data(data_dir+'/test.txt')

