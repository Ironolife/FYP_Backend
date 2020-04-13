import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import os
import random
import re

ARTICLE_LENGTH = 50
SUMMARY_LENGTH = 14
BATCH_SIZE = 64

def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

class data():

    def __init__(self, args):

        self.word2id = read_json('data/word2id.json')
        self.id2word = [[]] * len(self.word2id)
        for word in self.word2id:
            self.id2word[self.word2id[word]] = word

        self.vocab_size = len(self.word2id)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = stopwords.words('english')

    def prepare_data(self, args):

        if args.datatype == 'gigaword':
            self.prepare_data_gigaword()
        elif args.datatype == 'reuters':
            self.prepare_data_reuters()
        elif args.datatype == 'cnn':
            self.prepare_data_cnn()

    def prepare_data_gigaword(self):

        TEST_SIZE = 10048

        result = []

        article_file = open('./data/gigaword/article.txt', encoding="utf8")
        summary_file = open('./data/gigaword/summary.txt', encoding="utf8")
        processed_article_file = open('./data/gigaword/preprocessed_article.txt','w', encoding="utf8")
        processed_summary_file = open('./data/gigaword/preprocessed_summary.txt','w', encoding="utf8")
        test_article_file = open('./data/gigaword/test_article.txt', 'w', encoding="utf8")
        test_summary_file = open('./data/gigaword/test_summary.txt', 'w', encoding="utf8")

        line_count = 0;

        for article, summary in zip(article_file, summary_file):

            article = self.remove_stopwords_and_punctuation(article)
            summary = self.remove_stopwords_and_punctuation(summary)

            article.lower()
            summary.lower()

            result.append([article, summary])

            line_count+=1

        for i in range(line_count):

            if i < line_count - TEST_SIZE:
                processed_article_file.write(result[i][0] + '\n')
                processed_summary_file.write(result[i][1] + '\n')
            else:
                test_article_file.write(result[i][0] + '\n')
                test_summary_file.write(result[i][1] + '\n')

        print(str(line_count) + ' lines procecessed.')

    def prepare_data_reuters(self):

        result = []

        processed_article_file = open('./data/reuters/preprocessed_article.txt','w', encoding="utf8")
        processed_summary_file = open('./data/reuters/preprocessed_summary.txt','w', encoding="utf8")

        file_count = 0
        line_count = 0

        paragraphs = []

        for f in os.listdir('./data/reuters/raw'):
            with open('./data/reuters/raw/' + f, 'r', encoding="utf8") as raw_text:

                title = raw_text.readline()
                article = raw_text.readlines()
                paragraph = ''

                for i in range(len(article)):
                    if i == 0:
                        paragraph += self.remove_stopwords_and_punctuation(article[i].strip()).lower()
                    else:
                        if not article[i].startswith('      '):
                            paragraph += ' ' + self.remove_stopwords_and_punctuation(article[i].strip()).lower()
                        else:
                            if len(paragraph) > 40:
                                paragraphs.append(paragraph + '\n')
                            paragraph = self.remove_stopwords_and_punctuation(article[i].strip()).lower()

                if len(paragraph) > 40:
                    paragraphs.append(paragraph + '\n')

                file_count += 1
                print(file_count)

        for paragraph in paragraphs:

            words = paragraph.split()
            random_length = random.randint(6,11)
            random_start = random.randint(0,10)

            summary = []

            for i, word in enumerate(words):
                if word in self.word2id and word != ',':
                    if len(summary) < random_length and i >= random_start:
                        summary.append(word)

            if(len(summary) > 3):
                processed_summary_file.write(' '.join(summary) + '\n')
                processed_article_file.write(paragraph)
                line_count += 1

        print(str(file_count) + ' file procecessed. ' + str(line_count) + ' lines generated.')

    def prepare_data_cnn(self):

        result = []

        article_file = open('./data/cnn/raw/train.txt.src', encoding="utf8")
        summary_file = open('./data/cnn/raw/train.txt.tgt.tagged', encoding="utf8")
        processed_article_file = open('./data/cnn/preprocessed_article.txt','w', encoding="utf8")
        processed_summary_file = open('./data/cnn/preprocessed_summary.txt','w', encoding="utf8")

        line_count = 0

        for article, summary in zip(article_file, summary_file):

            article = article.replace('-lrb-', '').replace('-rrb-', '').replace('cnn', '').replace('--', '')
            summary = summary.replace('<t>', '').replace('</t>', '').replace('-lrb-', '').replace('-rrb-', '')

            article = self.remove_stopwords_and_punctuation(article)
            summary = self.remove_stopwords_and_punctuation(summary)

            article.lower()
            summary.lower()

            result.append([article, summary])

            line_count+=1

            if line_count % 1000 == 0:
                print(line_count)

        for i in range(line_count):

            processed_article_file.write(result[i][0] + '\n')
            processed_summary_file.write(result[i][1] + '\n')

        print(str(line_count) + ' lines procecessed for training.')

        test_result = []
        
        test_raw_article_file = open('./data/cnn/raw/test.txt.src', encoding="utf8")
        test_raw_summary_file = open('./data/cnn/raw/test.txt.tgt.tagged', encoding="utf8")
        test_article_file = open('./data/cnn/test_article.txt', 'w', encoding="utf8")
        test_summary_file = open('./data/cnn/test_summary.txt', 'w', encoding="utf8")

        test_line_count = 0

        for article, summary in zip(test_raw_article_file, test_raw_summary_file):

            article = article.replace('-lrb-', '').replace('-rrb-', '').replace('cnn', '').replace('--', '')
            summary = summary.replace('<t>', '').replace('</t>', '').replace('-lrb-', '').replace('-rrb-', '')

            article = self.remove_stopwords_and_punctuation(article)
            summary = self.remove_stopwords_and_punctuation(summary)

            article.lower()
            summary.lower()

            test_result.append([article, summary])

            test_line_count+=1

            if test_line_count % 1000 == 0:
                print(test_line_count)

        for i in range(test_line_count):

            test_article_file.write(test_result[i][0] + '\n')
            test_summary_file.write(test_result[i][1] + '\n')

        print(str(test_line_count) + ' lines procecessed for testing.')

    def remove_stopwords_and_punctuation(self, line):

        words = self.tokenizer.tokenize(line)
        [word for word in words if word not in self.stopwords]
        return ' '.join(words)

    def sentence2id(self, line, length):

        line = line.strip()
        ids = np.zeros((length), dtype=np.int32)
        
        i = 0
        for word in line.split():
            if word in self.word2id and word != '<unk>':
                ids[i] = self.word2id[word]
                i += 1
            if i >= length:
                break
        return ids

    def id2sentence(self, ids):

        sentence = []
        current = dict()

        for id in ids:

            if id in current:
                continue

            current[id] = 1
            
            if id <= 1:
                break

            sentence.append(self.id2word[id])

        return ' '.join(sentence)

    def sentence_generator(self, file):

        while True:
            for line in open(file):
                yield line.strip()

    def pretrain_generator(self, datatype):

        x_batch = []
        y_batch = []

        processed_article_path = './data/' + datatype + '/preprocessed_article.txt'
        processed_summary_path = './data/' + datatype + '/preprocessed_summary.txt'

        for article, summary in zip(self.sentence_generator(processed_article_path), self.sentence_generator(processed_summary_path)):

            x_ids = self.sentence2id(article, ARTICLE_LENGTH)
            y_ids = self.sentence2id(summary, SUMMARY_LENGTH)

            x_batch.append(x_ids)
            y_batch.append(y_ids)

            if len(x_batch) == BATCH_SIZE:

                yield np.array(x_batch), np.array(y_batch)
                
                x_batch = []
                y_batch = []

    def train_generator(self, datatype):

        x_batch = []
        y_batch = []

        processed_article_path = './data/' + datatype + '/preprocessed_article.txt'
        processed_summary_path = './data/' + datatype + '/preprocessed_summary.txt'

        for article, summary in zip(self.sentence_generator(processed_article_path), self.sentence_generator(processed_summary_path)):

            x_ids = self.sentence2id(article, ARTICLE_LENGTH)
            y_ids = self.sentence2id(summary, SUMMARY_LENGTH)

            x_batch.append(x_ids)
            y_batch.append(y_ids)

            if len(x_batch) == BATCH_SIZE:

                yield np.array(x_batch), np.array(y_batch)

                x_batch = []
                y_batch = []

    def test_generator(self):

        x_batch = []

        for article in open('./data/gigaword/test_article.txt'):

            x_batch.append(self.sentence2id(article, ARTICLE_LENGTH))

            if len(x_batch) == BATCH_SIZE:

                yield np.array(x_batch)

                x_batch = []

    def input_generator(self, input):

        x_batch = []

        for i in range(len(input)):

            x_batch.append(self.sentence2id(input[i], ARTICLE_LENGTH))

            if len(x_batch) == BATCH_SIZE:

                yield np.array(x_batch)

                x_batch = []

            if len(input) % BATCH_SIZE != 0 and i == len(input) - 1:

                while len(x_batch) != BATCH_SIZE:

                    x_batch.append(self.sentence2id('', ARTICLE_LENGTH))

                yield np.array(x_batch)