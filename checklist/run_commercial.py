import sys
sys.path.append('/home/marcotcr/work/ml-tests/')
from mltests import online_model_wrapper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model", required=True, choices=['google', 'microsoft', 'amazon'])
parser.add_argument('-i', "--input_file", required=True)
parser.add_argument('-o', "--output_file", required=True)
args = parser.parse_args()

if args.model == 'microsoft':
    model = online_model_wrapper.OnlinePredictor('/home/marcotcr/tmp/azure3_preds_cameraready.pkl', batch_size=1000, wait_time=1, model='azure3')
elif args.model == 'google':
    model = online_model_wrapper.OnlinePredictor('/home/marcotcr/tmp/google_preds_cameraready.pkl', batch_size=100, wait_time=20, model='google')
elif args.model == 'amazon':
    model = online_model_wrapper.OnlinePredictor('/home/marcotcr/tmp/amazon_preds_cameraready.pkl', batch_size=100, wait_time=0.1, model='amazon')

texts = open(args.input_file, 'r').read().splitlines()
preds, confs = model.predict_and_confidences(texts)
open(args.output_file, 'w').write('\n'.join(['%d %f %f %f' % (pred, *c) for pred, c in zip(preds, confs)]))
