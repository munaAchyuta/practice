import re, os
import math
from collections import defaultdict
from itertools import chain
import pandas as pd
import numpy as np
import nltk, string
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
import pickle
#nltk.download('punkt') # if necessary...

USE_NGRAMS = True
KEYWORDS_FILE_NAME = "headers_only_filtered_data.csv"
TEXT_FIELD_NAME = 'content'
LABEL_FIELD_NAME = 'label_content'
USE_TOP_FEATURES = True
DO_DOWN_SAMPLING = False
DO_UP_SAMPLING = True
USE_TOP_N_FEATURES = 30
USE_ML_MODEL = True
MODEL_PATH = './model/multiClass_classifier.pkl'

if not os.path.exists('./model'):
    os.makedirs('./model')

# temporary
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

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
        self.do_up_sampling = DO_UP_SAMPLING
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
        tmp_list = []
        for grm in ngrams_ind:
            tmp_list.extend(list(ngrams(list_of_words, grm)))
        
        return [" ".join(tmp_list_i) for tmp_list_i in tmp_list]
    
    def cleanse(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]+", ' ', text.strip())
        text = ''.join([i for i in text if not i.isdigit()])
        filtered_words = [w for w in nltk.word_tokenize(text.translate(self.remove_punctuation_map)) if w.lower() not in self.stop_words]
        filtered_words = [w for w in filtered_words if len(w)>1]
        
        return self.lemm_tokens(filtered_words)

    def normalize(self,text):
        lemma_tokens = self.cleanse(text)
        #lemma_tokens = self.lemm_tokens(text.split())
        
        if self.use_ml_model:
            return " ".join(lemma_tokens)
        
        if self.use_ngrams:
            return self.add_ngrams(lemma_tokens)
        return lemma_tokens
    
    def balance_samples(self, df):
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

        # Downsample majority class
        list_of_df_majority_downsampled = []
        df_below_avg = []
        for each_tup in listof_tuples:
            if each_tup[1] <= min_val:
                df_below_avg.append(each_tup[0])
                continue
            
            df_majority = df[df.category_id==each_tup[0]]
            df_majority_downsampled = resample(df_majority, replace=False, n_samples=min_val, random_state=123)
            list_of_df_majority_downsampled.append(df_majority_downsampled)
        
        for df_below_key in df_below_avg:
            list_of_df_majority_downsampled.append(df[df.category_id==df_below_key])

        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat(list_of_df_majority_downsampled)

        # Display new class counts
        print(df_downsampled.category_id.value_counts())
        return df_downsampled
    
    def over_sample(self, features, labels):
        smt = SMOTETomek(ratio='auto')
        X_smt, y_smt = smt.fit_sample(features, labels)
        
        return X_smt, y_smt

    def get_label_keywords(self):
        # load train/test dataset
        df = pd.read_csv(self.keywords_file_name)
        df['category_id'] = df[self.label_field_name].factorize()[0]
        
        category_id_df = df[[self.label_field_name, 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', self.label_field_name]].values)
        
        # balance samples.
        if self.do_down_sampling:
            df = self.balance_samples(df)
        
        self.class_content_dict = defaultdict(list)
        
        if self.use_top_features:
            self.use_ngrams = True
        
        #if not self.use_top_features:
        grouped = df.groupby(self.label_field_name)
            
        for name, group in grouped:
            self.class_content_dict[name].extend(group[self.text_field_name].tolist())
            
        all_unique_terms = []
        for key in self.category_to_id.keys():
            self.class_content_dict[key] = [self.normalize(text) for text in self.class_content_dict[key]]
            class_keys = list(chain(*self.class_content_dict[key]))
            self.class_content_dict[key] = FreqDist(class_keys)
            all_unique_terms.extend(class_keys)
        all_unique_terms = list(set(all_unique_terms))
        
        # normalize common/rare words by idf.
        term_idf_dict = {}
        total_documents_cnt = len(self.category_to_id)
        for each_term in all_unique_terms:
            term_in_docs = [True for each_class_keys, each_class_vals in self.class_content_dict.items() if each_term in each_class_vals]
            term_in_docs_cnt = 1 if term_in_docs.count(True)==0 else term_in_docs.count(True)
            term_idf = round(math.log10(total_documents_cnt / term_in_docs_cnt), 3)
            term_idf_dict[each_term] = term_idf
        
        # do tf*idf and save in self.class_content_dict
        for key in self.category_to_id.keys():
            self.class_content_dict[key] = {key:val*term_idf_dict[key] for key,val in self.class_content_dict[key].items()}
            
        
        if self.use_top_features:
            self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')
            self.tokenizer = self.tfidf.build_tokenizer()
            self.preprocess = self.tfidf.build_preprocessor()
            
            df[self.text_field_name] = df[self.text_field_name].apply(lambda x: " ".join(self.cleanse(x)))
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
                
                filtered_dict = {key:val for key,val in self.class_content_dict[label_key].items() if key in unigrams[-N:]+bigrams[-N:]+trigrams[-N:]}
                self.class_content_dict[label_key] = filtered_dict
        #
        #print(self.class_content_dict)
    
    def define_model(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')
        self.tokenizer = self.tfidf.build_tokenizer()
        self.preprocess = self.tfidf.build_preprocessor()
        self.model = LinearSVC()
    
    def train_model(self):
        # load train/test dataset
        df = pd.read_csv(self.keywords_file_name)
        df['category_id'] = df[self.label_field_name].factorize()[0]
        df[self.text_field_name] = df[self.text_field_name].apply(lambda x: self.normalize(x))
        target_label_names = df[self.label_field_name].unique()
            
        category_id_df = df[[self.label_field_name, 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', self.label_field_name]].values)
        
        # balance samples. down sample.
        if self.do_down_sampling:
            df = self.balance_samples(df)
        
        features = self.tfidf.fit_transform(df[self.text_field_name]).toarray()
        labels = df.category_id
        
        # over sample if not down sample.
        if self.do_up_sampling:
            features, labels = self.over_sample(features, labels)
            df = pd.DataFrame(features)
        
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(metrics.classification_report(y_test, y_pred, target_names=target_label_names))
        
        # Save the trained model as a pickle string.
        with open(self.model_path, 'wb') as f:
             pickle.dump((self.tfidf, self.model), f)
    
    def load_model(self):
        # load train/test dataset
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

    def similarity(self, words_l, words_r):
        words_l = set(words_l)
        words_match_count_score = round(sum([words_r.get(word,0) for word in words_l if word in words_r]) / len(words_l),3)

        return words_match_count_score
        #return round(len(words_l.intersection(words_r)) / len(words_l),3)

    def get_content_type(self, text):
        if self.use_ml_model:
            return self.id_to_category.get(self.model.predict(self.tfidf.transform([text]).toarray())[0],'o')
        
        labels_dict = {key:0.0 for key in self.category_to_id.keys()}
        for key in self.category_to_id.keys():
            labels_dict[key] = self.similarity(text, self.class_content_dict[key])

        max_val = 0.0
        max_item = 'o'
        for item in sorted(labels_dict.keys(), reverse = True):
            if labels_dict[item]>max_val:
                max_val = labels_dict[item]
                max_item = item
        
        return max_item

if __name__=="__main__":
    header_texts = ['TECHNOLOGY', 'Contact Information:', 'Customer', 'Information:']
    obj = ContentType()
    normalized_texts = [obj.normalize(text) for text in header_texts]
    texts_with_content_type = [obj.get_content_type(each_normalized_text) for each_normalized_text in normalized_texts]
    for i,j,k in zip(header_texts,normalized_texts,texts_with_content_type):
        #print(i,"--->",j,"--->",k)
        print(i,"--->",k)
