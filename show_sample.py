import sys
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFont, ImageDraw
args=sys.argv
vit_result=open('output/vit_only/pred.txt','r').read().strip().split('\n')
bert_result=open('output/bert_only/pred.txt','r').read().strip().split('\n')
bert_vit_result=open('output/bert_vit/pred.txt','r').read().strip().split('\n')
true_result=open('output/bert_vit/true.txt','r').read().strip().split('\n')
test_dataset=open('data/test.txt','r').read().strip().split('\n')
img_dir='../multimodal-sarcasm/data/dataset_image/'

total=list(zip(vit_result, bert_result, bert_vit_result, true_result, test_dataset))
random.shuffle(total)
for vr, br, bvr, tr, d in total:
	if not (vr==tr and tr!=br and tr!=bvr):
		continue
	plt.subplot(1,2,1)
	dd=eval(d)
	img=img_dir+dd[0]+".jpg"
	im=Image.open(img)
	plt.imshow(im)

	plt.subplot(1,2,2)
	plt.plot(0,0)
	plt.xlim((-5,5))
	plt.ylim((-5,5))

	plt.text(-4, 4, dd[1],fontdict={'size': 10, 'color': 'r'},wrap=True)
	plt.text(0, -1, str(tr),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -2, str(bvr),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -3, str(br),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -4, str(vr),fontdict={'size': 16, 'color': 'r'})
	
	plt.show()
