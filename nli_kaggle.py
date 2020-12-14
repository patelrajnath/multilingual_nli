"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.
It does NOT produce a sentence embedding and does NOT work for individual sentences.
Usage:
python training_nli.py
"""
import pandas
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import numpy as np
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# As dataset, we use SNLI + MultiNLI
# Check if dataset exsist. If not, download and extract  it
dataset_path = 'contradictory-my-dear-watson/train.csv'

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
dev_samples = []
df = pandas.read_csv(dataset_path)
for id, row in df.iterrows():
    label_id = int(row['label'])
    train_samples.append(InputExample(texts=[row['premise'], row['hypothesis']], label=label_id))

train_batch_size = 16
num_epochs = 10
model_save_path = 'output/training_allnli-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
# model = CrossEncoder('sentence-transformers/distilbert-base-nli-stsb-mean-tokens', num_labels=len(label2int))
# model = CrossEncoder('sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
#                      num_labels=len(label2int))
# model = CrossEncoder('sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens', num_labels=len(label2int))
model = CrossEncoder('joeddav/xlm-roberta-large-xnli', num_labels=len(label2int))

# We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='AllNLI-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

test_dataset = 'contradictory-my-dear-watson/test.csv'
df = pandas.read_csv(test_dataset)
sentence_pairs = []
ids = []
for id, row in df.iterrows():
    label_id = 0
    ids.append(row['id'])
    sentence_pairs.append([row['premise'], row['hypothesis']])

pred_scores = model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
pred_labels = np.argmax(pred_scores, axis=1)

out_df = pandas.DataFrame([ids, pred_labels]).transpose()
out_df = out_df.rename(columns={0: 'id', 1: 'prediction'})
out_df.to_csv('submission.csv', index=False)
