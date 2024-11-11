from sklearn.metrics import precision_recall_fscore_support
import sys


argv=sys.argv

preds=open(argv[1], 'r').read().strip().split('\n')
trues=open(argv[2], 'r').read().strip().split('\n')

print(precision_recall_fscore_support(trues, preds, average='macro'))
