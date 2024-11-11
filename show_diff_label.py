import sys
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFont, ImageDraw
args=sys.argv
test_dataset=open('../baseline/datasets/sarcasm/test.txt','r')
re1=open('../baseline_plus_ocr_v4/output/bert_vit/pred.txt', 'r').read().strip().split('\n')
re2=open('../baseline_plus_ocr_v5/output/bert_vit/pred.txt', 'r').read().strip().split('\n')
re3=open('../baseline_plus_ocr_v2/output/bert_vit/pred.txt', 'r').read().strip().split('\n')
img_dir='../dataset_image/'
c=0
for line in test_dataset:
	a=eval(line)
	r1=re1[c];
	r2=re2[c];
	r3=re3[c];
	c+=1

	if a[2]==a[3]:
		continue
	print(a[0])
	plt.subplot(1,2,1)
	img=img_dir+a[0]+".jpg"
	im=Image.open(img)
	plt.imshow(im)

	plt.subplot(1,2,2)
	plt.plot(0,0)
	plt.xlim((-5,5))
	plt.ylim((-5,5))

	plt.text(-4, 4, a[1],fontdict={'size': 10, 'color': 'r'},wrap=True)
	plt.text(0, -1, str(a[2]),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -2, str(a[3]),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -3, str(r1),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -4, str(r2),fontdict={'size': 16, 'color': 'r'})
	plt.text(0, -5, str(r3),fontdict={'size': 16, 'color': 'r'})
	
	plt.show()
