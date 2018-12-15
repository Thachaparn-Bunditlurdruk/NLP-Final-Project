# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 23:06:48 2018

@author: Administrator
"""
import string
import pandas
import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import word_tokenize

class Data_processing():
    def __init__(self):         
        pass
    
    @classmethod
    def tokenizer(self, text: str):
        """tokenize text
        
        Return: a dictionary with two keys which are a list of lexical word 
                and a list of punctuation(i.e. '.', ',', '-', '?', '!')
        
        >>> Data_processing.tokenizer("More Brexit assurances possible, says May")
        '{'punct': [','], 
        'word' : ['More', 'Brexit', 'assurance', 'possible', 'says' ,'May']}
        """
        
        token_dict = {'punct' : [], 'word' : []}
        tokenized_text = word_tokenize(text)
        considered_punct = '.,-?!'
        for token in tokenized_text:
            if token in considered_punct:
                token_dict['punct'].append(token)
            elif token not in string.punctuation:
                token_dict['word'].append(token)
        return token_dict
    
    @classmethod
    def lexical_diversity(self, tokeninzed_text: list):
        """count word type and word token
        
        Return: a ratio of word token and  word type
        
        >>> Data_processing.lexical_diversity(['And', 'what', 'came', 'out', 'of', 
        'that', 'was', 'his', 'clarity', 'is', 'that', 'actually', 'right'])
        0.9231
        """
        
        word_token = len(tokeninzed_text)
        word_type = len(set(tokeninzed_text))
        ratio = round(word_type/word_token, 4)
        return ratio
    
    @classmethod
    def bigram_generator(self, tokeninzed_text: list):
        """create bigram
        
        Return: a list of bigram
        
        >>> Data_processing.lexical_diversity(['More', 'Brexit', 'assurance', 
        'possible', ',', 'says' ,'May'])
        ['More Brexit', 'Brexit assurance', 'assurance possible', 'possible ,',
         ', says']
        """
        
        bigram_list = []
        for i in range(len(tokeninzed_text)):
            if i+2 < len(tokeninzed_text):
                bigram = '{} {}'.format(tokeninzed_text[i], tokeninzed_text[i+1])
                bigram_list.append(bigram)
        return bigram_list

    @classmethod
    def get_feature(self, dataset: list):
        """extract 6 features from text: punctuation feature, word feature, 
        readability feature, lexical diversity, url feature, and bigram feature
        
        Return: a list of feature dictionary
        
        >>> Data_processing.get_feature(data)
        [{'Maria struck': 1, 'struck': 1, 'www.bbc.com': 1, 'lexical_diversity' : 0.34}]
        """
        
        # generate a list for feature in dataset
        list_of_feature_dict = []
        
        # iterate on the list that its element are tuples
        for url, hl, text, label in dataset:
            
            # tokenize into 2 categories: word and considered punctuation
            # tokenize news headline
            dict_token_hl = self.tokenizer(hl)
            list_words_hl = dict_token_hl['word']
            list_punct_hl = dict_token_hl['punct']
            # tokenize text in body of a news
            dict_token_text = self.tokenizer(text)
            list_words_text = dict_token_text['word']
            list_punct_text = dict_token_text['punct']

            # generate a dictionary for features of each news
            feature_dict = {}
            
            # generate punctuation feature
            for p in list_punct_hl:
                feature_dict[p] = 1
            for p in list_punct_text:
                feature_dict[p] = 1

            # generate word feature
            for word in list_words_hl:
                feature_dict[word] = 1
            for word in list_words_text:
                feature_dict[word] = 1

            # generate readability features: a number of character and a number of paragraph
            feature_dict['num_character'] = len(hl + text)
            feature_dict['paragraph_count'] = text.count('\n')

            # generate lexical diversity feature
            feature_dict['diversity_hl'] = self.lexical_diversity(list_words_hl)
            feature_dict['diversity_body'] = self.lexical_diversity(list_words_text)

            # generate url feature to keep the source of news
            regex = re.compile(r'((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?')
            for match in regex.finditer(url):
                agency = match.group(3)
                feature_dict[agency] = 1

            # generate bigram feature
            token_hl = word_tokenize(hl)
            token_body = word_tokenize(text)
            bigram_hl = self.bigram_generator(token_hl)
            bigram_body = self.bigram_generator(token_body)
            for bi_hl in bigram_hl:
                feature_dict[bi_hl] = 1
            for bi_body in bigram_body:
                feature_dict[bi_body] = 1

            # generate a list of feature dictionary 
            list_of_feature_dict.append(feature_dict)
        return list_of_feature_dict
        
class DetectFakeNews(Data_processing):
    
    def __init__(self):
        # read file
        dict_file = pandas.read_csv('data.csv',  encoding="utf8")
        raw_data = [(url, hl, text, label) for url, hl, text, label in zip(dict_file['URLs'], 
                    dict_file['Headline'], dict_file['Body'], dict_file['Label']) 
                    if not pandas.isna(text)]
        self.data_train = [raw_data[n] for n in range(len(raw_data)) if n < len(raw_data)*80/100]
        self.data_test = [raw_data[n] for n in range(len(raw_data)) if n > len(raw_data)*80/100]
        
        # for trainning
        self.model = RandomForestClassifier(n_estimators= 600, min_samples_split= 5, 
                                            min_samples_leaf= 1, max_features= 'sqrt', 
                                            max_depth= 60, bootstrap= True)
        self.dict_vectorizer = DictVectorizer(sparse=True)
        self.train(self.model)  # to call function train() while instantiate

    def train(self, model: RandomForestClassifier):
        """Train fake news detector

        Open the news dataset and train the model
        """
        
        # prepare parameter for training model
        train_list_of_feature_dict = self.get_feature(self.data_train)
        train_label_list = [label for url, hl, text, label in self.data_train]

        # transform dense matrix of a list of feature_dict into sparse matrix
        train_sparse_feature_matrix = self.dict_vectorizer.fit_transform(train_list_of_feature_dict)
        
        # train model
        self.model = model.fit(train_sparse_feature_matrix, train_label_list)
        pass
    
    def detect(self, url: str, headline: str, body: str):
        """predict news whether it is fake or not
        
        Return: one of 'LEGITIMATE !!!' and 'FAKE !!!'
        
        >>> DetectFakeNews.detect('http://www.bbc.com/news/world-us-canada-41419190', 
        'Four ways Bob Corker skewered Donald Trump', "Image copyright Getty 
        Images On Sunday morning...")
        
        'LEGITIMATE !!!'
        """
        label = None
        dataset = [(url, headline, body, label)]
        
        # prepare parameter for prediction
        test_list_of_feature_dict = self.get_feature(dataset)
        test_sparse_feature_matrix = self.dict_vectorizer.transform(test_list_of_feature_dict)  # generate tested sparse feature matrix
        
        #prediction
        predicted_value = self.model.predict(test_sparse_feature_matrix)
        
        # label_interpret = {1 : 'LEGITIMATE', 0 : 'FAKE'}
        if predicted_value == 1:
            return print('LEGITIMATE !!!')
        elif predicted_value == 0:
            return print('FAKE !!!')
   
    def evaluation(self):
        """
        evaluate model's performance
        
        Return: accuracy, precision, recall, f1
        
        >>> DetectFakeNews.evaluate()
        accuracy = 0.9943520552243489 
        precision, recall, f1 = (0.9944189387809027, 0.9943520552243489, 0.9943537111789301, None)
        """
        # prepare parameter of testing data
        test_label_list =  [label for url, hl, text, label in self.data_test]
        test_list_of_feature_dict = self.get_feature(self.data_test)
        test_sparse_feature_matrix = self.dict_vectorizer.transform(test_list_of_feature_dict)  # generate tested sparse feature matrix
        
        # prediction
        predicted_value = self.model.predict(test_sparse_feature_matrix)  # predict value from trained model
        
        # accuracy
        accuracy = self.model.score(test_sparse_feature_matrix, test_label_list)
        
        # precision recall f1
        y_true = np.array(test_label_list)
        y_pred = np.array(predicted_value)
        evaluate = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return print("""accuracy = {} \nprecision, recall, f1 = {}""".format(accuracy, evaluate))