import re, os
from collections import defaultdict
from itertools import chain
import pandas as pd
import numpy as np
import nltk, string
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
import pickle
#nltk.download('punkt') # if necessary...

USE_NGRAMS = True
KEYWORDS_FILE_NAME = "data.csv" # should contain two columns. 1.text & 2.label
TEXT_FIELD_NAME = 'content'
LABEL_FIELD_NAME = 'label_content'
USE_TOP_FEATURES = True
DO_DOWN_SAMPLING = True
USE_TOP_N_FEATURES = 20
USE_ML_MODEL = False
MODEL_PATH = './model/multiClass_classifier.pkl'

if not os.path.exists('./model'):
    os.makedirs('./model')

class ContentType(object):
    '''
    class for getting content-type by doing keyword match or using ML-algo(linear-svc with TF-IDF vectorizer).
    Default keyword based match result will come without including ngrams. to include ngrams , user has to set USE_NGRAMS = True.
    if user wants to to select to most corelated words per class, then make USE_TOP_FEATURES = True and choose number of keywords to be selected for in top by `USE_TOP_N_FEATURES = 20`. here CHI-Square algo used using TF-IDF scores for selecting top N keywords.
    choose `USE_ML_MODEL = True` for use of ML-model else choose `USE_ML_MODEL = False`
    '''
    def __init__(self):
        self.use_ngrams = USE_NGRAMS
        self.text_field_name = TEXT_FIELD_NAME
        self.label_field_name = LABEL_FIELD_NAME
        self.keywords_file_name = KEYWORDS_FILE_NAME
        self.do_down_sampling = DO_DOWN_SAMPLING
        self.use_top_features = USE_TOP_FEATURES
        self.use_top_n_features = USE_TOP_N_FEATURES
        self.use_ml_model = USE_ML_MODEL
        self.model_path = MODEL_PATH
        self.use_lower = True
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        self.stop_words = set(stopwords.words('english'))
        if not self.use_ml_model:
            self.get_label_keywords()
        else:
            if os.path.exists(self.model_path):
                print("load model")
                self.load_model()
            else:
                print("train model")
                self.define_model()
                self.train_model()

    def lemm_tokens(self, tokens):
        return [self.wordnet_lemmatizer.lemmatize(item) for item in tokens]
    
    def add_ngrams(self, list_of_words, ngrams_ind=[1,2,3]):
        '''
        input : list_of_words-->type(list)
        generate ngrams.
        '''
        tmp_list = []
        for grm in ngrams_ind:
            tmp_list.extend(list(ngrams(list_of_words, grm)))
        return [" ".join(tmp_list_i) for tmp_list_i in tmp_list]
    
    def cleanse(self, text):
        '''
        remove punctuation/digits, lowercase, lemmatize
        '''
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]+", ' ', text.strip())
        text = ''.join([i for i in text if not i.isdigit()])
        filtered_words = [w for w in nltk.word_tokenize(text.translate(self.remove_punctuation_map)) if w.lower() not in self.stop_words]
        return " ".join(filtered_words)

    def normalize(self,text):
        text = self.cleanse(text)
        lemma_tokens = self.lemm_tokens(text.split())
        
        if self.use_ml_model:
            return " ".join(lemma_tokens)
        
        if not self.use_top_features:
            if self.use_ngrams:
                return self.add_ngrams(lemma_tokens)
            return lemma_tokens
        return self.get_most_corelated_words(" ".join(lemma_tokens))
    
    def balance_samples(self, df):
        '''
        down-sample
        '''
        # Display old class counts
        print(df.category_id.value_counts())
        
        # Separate majority and minority classes
        class_df_count = {key:0 for key in self.id_to_category.keys()}
        for each_class in self.id_to_category.keys():
            df_class = df[df.category_id==each_class]
            class_df_count[each_class] = len(df_class)
        
        listof_tuples = sorted(class_df_count.items() ,  key=lambda x: x[1])
        min_val = listof_tuples[0][1]
        min_item = listof_tuples[0][0]
        avg_val = int(sum(class_df_count.values()) / len(class_df_count))
        max_val = listof_tuples[-1][1]
        max_item = listof_tuples[-1][0]

        list_of_df_majority_downsampled = []
        # Downsample majority class
        df_below_avg = []
        for each_tup in listof_tuples:#listof_tuples[1:]:
            if each_tup[1] <= min_val:
                df_below_avg.append(each_tup[0])
                continue
            df_majority = df[df.category_id==each_tup[0]]
            df_majority_downsampled = resample(df_majority, 
                                             replace=False,    # sample without replacement
                                             n_samples=min_val,     # to match minority class
                                             random_state=123) # reproducible results
            list_of_df_majority_downsampled.append(df_majority_downsampled)
        
        for df_below_key in df_below_avg:
            list_of_df_majority_downsampled.append(df[df.category_id==df_below_key])

        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat(list_of_df_majority_downsampled)

        # Display new class counts
        print(df_downsampled.category_id.value_counts())
        return df_downsampled

    def get_label_keywords(self):
        '''
        based on your label, change keys of class_content_dict and get desired variables.
        '''
        df = pd.read_csv(self.keywords_file_name)
        df['category_id'] = df[self.label_field_name].factorize()[0]
        
        category_id_df = df[[self.label_field_name, 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', self.label_field_name]].values)
        
        # balance samples.
        if self.do_down_sampling:
            df = self.balance_samples(df)
        
        self.class_content_dict = defaultdict(list)
        
        if not self.use_top_features:
            grouped = df.groupby(self.label_field_name)
            
            for name, group in grouped:
                self.class_content_dict[name].extend(group[self.text_field_name].tolist())
            
            for key in self.category_to_id.keys():
                self.class_content_dict[key] = [self.normalize(text) for text in self.class_content_dict[key]]
                self.class_content_dict[key] = set(chain(*self.class_content_dict[key]))
        else:
            df[self.text_field_name] = df[self.text_field_name].apply(lambda x: " ".join(self.lemm_tokens(self.cleanse(x).split())))
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')
            self.tokenizer = self.tfidf.build_tokenizer()
            self.preprocess = self.tfidf.build_preprocessor()
            features = self.tfidf.fit_transform(df[self.text_field_name]).toarray()
            labels = df.category_id
            
            N = self.use_top_n_features
            for label_key, category_id in sorted(self.category_to_id.items()):
                features_chi2 = chi2(features, labels == category_id)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(self.tfidf.get_feature_names())[indices]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
                
                self.class_content_dict[label_key].extend(unigrams[-N:]+bigrams[-N:]+trigrams[-N:])
                self.class_content_dict[label_key] = set(self.class_content_dict[label_key])
        #
        #print(self.class_content_dict)
    
    def define_model(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')
        self.tokenizer = self.tfidf.build_tokenizer()
        self.preprocess = self.tfidf.build_preprocessor()
        self.model = LinearSVC()
    
    def train_model(self):
        # train model
        df = pd.read_csv(self.keywords_file_name)
        df['category_id'] = df[self.label_field_name].factorize()[0]
        df[self.text_field_name] = df[self.text_field_name].apply(lambda x: " ".join(self.lemm_tokens(self.cleanse(x).split())))
            
        category_id_df = df[[self.label_field_name, 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', self.label_field_name]].values)
        
        # balance samples.
        if self.do_down_sampling:
            df = self.balance_samples(df)
        
        features = self.tfidf.fit_transform(df[self.text_field_name]).toarray()
        labels = df.category_id
        
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(metrics.classification_report(y_test, y_pred, target_names=df[self.label_field_name].unique()))
        
        # Save the trained model as a pickle string.
        with open(self.model_path, 'wb') as f:
             pickle.dump((self.tfidf, self.model), f)
    
    def load_model(self):
        df = pd.read_csv(self.keywords_file_name)
        df['category_id'] = df[self.label_field_name].factorize()[0]
        category_id_df = df[[self.label_field_name, 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', self.label_field_name]].values)
        
        # load the trained model.
        with open(self.model_path, 'rb') as f:
            self.tfidf, self.model = pickle.load(f)
            self.tokenizer = self.tfidf.build_tokenizer()
            self.preprocess = self.tfidf.build_preprocessor()
    
    def get_most_corelated_words(self, text):
        '''
        this is just to follow sklearn tokenizer and ngram approach. actually have followed same approach in cleanse->normalise
        methods. so this one not needed.
        '''
        return self.tfidf._word_ngrams(self.tokenizer(self.preprocess(text)))#, stop_words=self.stop_words

    def similarity(self, words_l, words_r):
        '''
        input : words_l-->type(list)
        input : words_r-->type(list)
        get similarity score between two text lists.
        do it, by matching number of words from left with right and then devide it with total number of words from left.
        '''
        words_l = set(words_l)
        return round(len(words_l.intersection(words_r)) / len(words_l),3)

    def get_content_type(self, text):
        '''
        input : text-->type(string)
        based on your label, change labels_dict and dependency variables.
        '''
        if self.use_ml_model:
            return self.id_to_category.get(self.model.predict(self.tfidf.transform([text]).toarray())[0],'None')
        
        labels_dict = {key:0.0 for key in self.category_to_id.keys()}
        for key in self.category_to_id.keys():
            labels_dict[key] = self.similarity(text, self.class_content_dict[key])

        max_val = 0.0
        max_item = 'None'
        for item in sorted(labels_dict.keys(), reverse = True):
            if labels_dict[item]>max_val:
                max_val = labels_dict[item]
                max_item = item
        return max_item

if __name__=="__main__":
    header_texts = ['TECHNOLOGY', 'Contact Information:', 'Customer', 'Information:']
    obj = ContentType()
    '''
    to see top keywords which chi2 picked, just call obj.service_type_keywords and similarly for others.
    '''
    normalized_texts = [obj.normalize(text) for text in header_texts]
    texts_with_content_type = [obj.get_content_type(each_normalized_text) for each_normalized_text in normalized_texts]
    for i,j,k in zip(header_texts,normalized_texts,texts_with_content_type):
        #print(i,"--->",j,"--->",k)
        print(i,"--->",k)
