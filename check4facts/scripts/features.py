import os
import glob
import string
import csv

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords

from check4facts.config import DirConf


class FeaturesExtractor:

    def __init__(self, **kwargs):
        self.basic_params = kwargs['basic']
        self.emb_params = kwargs['embeddings']
        self.sim_params = kwargs['similarity']
        self.subj_params = kwargs['subjectivity']
        self.sent_params = kwargs['sentiment']
        self.emo_params = kwargs['emotion']

        self.nlp = spacy.load(self.basic_params['model'])
        self.lexicon_ = None
        csv.field_size_limit(self.basic_params['field_size_limit'])

    @property
    def lexicon(self):
        if self.lexicon_ is None:
            self.lexicon_ = pd.read_csv(self.basic_params['lexicon'], sep='\t')
            self.lexicon_ = self.lexicon_.fillna('N/A')
            self.lexicon_['Lemma'] = self.lexicon_['Term'].apply(
                lambda x: self.nlp(x.lower().split()[0])[0].lemma_)

            for feat in ['subj', 'sent', 'emo']:
                params = getattr(self, feat + '_params')
                cols = [prefix + str(i) for prefix in params['prefixes']
                        for i in range(1, 5)]
                for col in cols:
                    self.lexicon_[col] = self.lexicon_[col].map(params['scores'])
        return self.lexicon_

    @staticmethod
    def text_preprocess(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = ' '.join([word for word in text.split() if
                         word not in stopwords.words('greek')])
        return text

    @staticmethod
    def get_embedding(sent_doc):
        return sent_doc.vector

    def get_similarity(self, sent_doc, claim):
        claim_doc = self.nlp(self.text_preprocess(claim))
        return sent_doc.similarity(claim_doc)

    def get_subjectivity(self, annots):
        cols = [col for col in self.lexicon if col.startswith('Subjectivity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        return np.mean(scores)

    def get_subjectivity_c(self, annots):
        cols = [col for col in self.lexicon if col.startswith('Subjectivity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        obj_tokens = sum(s < self.subj_params['thr']['OBJ'] for s in scores)
        subj_tokens = sum(s > self.subj_params['thr']['SUBJ'] for s in scores)
        return obj_tokens, subj_tokens

    def get_sentiment(self, annots):
        cols = [col for col in self.lexicon if col.startswith('Polarity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        return np.mean(scores)

    def get_sentiment_c(self, annots):
        cols = [col for col in self.lexicon if col.startswith('Polarity')]
        scores = [np.mean(a[cols].values) if not a.empty else 0.5 for a in annots]
        neg_tokens = sum(s < self.sent_params['thr']['NEG'] for s in scores)
        pos_tokens = sum(s > self.sent_params['thr']['POS'] for s in scores)
        return neg_tokens, pos_tokens

    def get_emotion(self, annots):
        emotions = {}
        for emotion in self.emo_params['prefixes']:
            cols = [col for col in self.lexicon if col.startswith(emotion)]
            scores = [np.mean(a[cols].values) if not a.empty else 0.0 for a in annots]
            emotions[emotion] = np.min(scores), np.mean(scores), np.max(scores)
        # mean_values = [v[1] for k, v in emotions.items()]
        # emotions['argmax'] = np.argmax(mean_values)
        # emotions['mean'] = np.mean(mean_values)
        return emotions

    def aggregate_features(self, feats_list):
        aggr_feats = {}

        if 'embedding' in self.basic_params['included_feats']:
            feats = [d['embedding'] for d in feats_list]
            aggr_feats['embedding'] = np.mean(feats, axis=0)
        if 'similarity' in self.basic_params['included_feats']:
            feats = [d['similarity'] for d in feats_list]
            aggr_feats['similarity'] = np.mean(feats, axis=0)
        if 'subjectivity' in self.basic_params['included_feats']:
            feats = [d['subjectivity'] for d in feats_list]
            aggr_feats['subjectivity'] = np.mean(feats, axis=0)
        if 'subjectivity_c' in self.basic_params['included_feats']:
            feats = [d['subjectivity_c'] for d in feats_list]
            aggr_feats['subjectivity_c'] = tuple(np.mean(feats, axis=0))
        if 'sentiment' in self.basic_params['included_feats']:
            feats = [d['sentiment'] for d in feats_list]
            aggr_feats['sentiment'] = np.mean(feats, axis=0)
        if 'sentiment_c' in self.basic_params['included_feats']:
            feats = [d['sentiment_c'] for d in feats_list]
            aggr_feats['sentiment_c'] = tuple(np.mean(feats, axis=0))
        if 'emotion' in self.basic_params['included_feats']:
            aggr_feats['emotion'] = {}
            for emotion in self.emo_params['prefixes']:
                feats = [d['emotion'][emotion] for d in feats_list]
                min_ = np.min(feats, axis=0)[0]
                avg_ = np.mean(feats, axis=0)[1]
                max_ = np.max(feats, axis=0)[2]
                aggr_feats['emotion'][emotion] = min_, avg_, max_
        return aggr_feats

    def get_sentence_features(self, sent, claim):
        sent_doc = self.nlp(self.text_preprocess(sent)[:self.nlp.max_length])
        annots = [self.lexicon[self.lexicon['Lemma'] == t.lemma_] for t in sent_doc]
        feats = {}

        if 'embedding' in self.basic_params['included_feats']:
            feats['embedding'] = self.get_embedding(sent_doc)
        if 'similarity' in self.basic_params['included_feats']:
            feats['similarity'] = self.get_similarity(sent_doc, claim)
        if 'subjectivity' in self.basic_params['included_feats']:
            feats['subjectivity'] = self.get_subjectivity(annots)
        if 'subjectivity_c' in self.basic_params['included_feats']:
            feats['subjectivity_c'] = self.get_subjectivity_c(annots)
        if 'sentiment' in self.basic_params['included_feats']:
            feats['sentiment'] = self.get_sentiment(annots)
        if 'sentiment_c' in self.basic_params['included_feats']:
            feats['sentiment_c'] = self.get_sentiment_c(annots)
        if 'emotion' in self.basic_params['included_feats']:
            feats['emotion'] = self.get_emotion(annots)  # TODO decide map value for N/A
        return feats

    def get_resource_features(self, title, body, sim_par, sim_sent, claim):
        feats = {}

        if 'title' in self.basic_params['included_res_parts']:
            feats['title'] = self.get_sentence_features(title, claim)
        if 'body' in self.basic_params['included_res_parts']:
            pars_feats = [self.get_sentence_features(par, claim) for par in body.split('\n')]
            feats['body'] = self.aggregate_features(pars_feats)
        if 'sim_par' in self.basic_params['included_res_parts']:
            feats['sim_par'] = self.get_sentence_features(sim_par, claim)
        if 'sim_sent' in self.basic_params['included_res_parts']:
            feats['sim_sent'] = self.get_sentence_features(sim_sent, claim)
        return feats

    def get_claim_features(self, claim, input_file):
        feats = {'claim': None, 'resources': None}

        with open(input_file, newline='') as input_csv:
            reader = csv.DictReader(input_csv)
            res_feats = [self.get_resource_features(
                row['title'], row['body'], row['similar_paragraph'],
                row['similar_sentence'], row['claim_text']) for row in reader
                if row['title'] != '' and row['body'] != '']

        feats['claim'] = self.get_sentence_features(claim, claim)
        if res_feats:
            feats['resources'] = {res_part: self.aggregate_features(
                [d[res_part] for d in res_feats]) for res_part in
                self.basic_params['included_res_parts']}
        return feats

    def run(self):
        if not os.path.exists(DirConf.FEATURES_RESULTS_DIR):
            os.mkdir(DirConf.FEATURES_RESULTS_DIR)

        claims_df = pd.read_csv(os.path.join(DirConf.DATA_DIR, 'claims.csv'))
        path = os.path.join(DirConf.DATA_DIR, self.basic_params['dir_name'])
        files = glob.glob(os.path.join(path, '*.csv'))

        for file in files[:5]:
            print('Processing file:', file)
            claim_id = int(os.path.basename(file).split('.')[0])
            claim = claims_df.loc[claims_df['Fact id'] == claim_id, 'Text'].iloc[0]
            claim_feats = self.get_claim_features(claim, file)
            # print(claim_feats)

        # for key, value in all_feats.items():
        #     path = os.path.join(DirConf.FEATURES_RESULTS_DIR, key)
        #     np.save(path, np.vstack(value))
        return