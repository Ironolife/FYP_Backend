import argparse
import tensorflow as tf
from model import model
from data import data

def parse():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('--datatype')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()
    return args

args = parse()

if args.action == 'data':
    if args.datatype != 'gigaword' and args.datatype != 'reuters' and args.datatype != 'cnn':
        print('Invalid data type.')
    else:
        data = data(args)
        data.prepare_data(args)
else:
    sess = tf.Session()
    model = model(sess, args)
    if(args.action == 'pretrain'):
        model.pretrain()
    elif(args.action == 'train'):
        model.train()
    elif(args.action == 'test'):
        model.test()
    elif(args.action == 'save'):
        model.save()