import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from text_utils import preprocess


class BM25Gensim:
    def __init__(self, checkpoint_path, entity_dict, title2idx):
        self.dictionary = Dictionary.load(checkpoint_path + "/dict")
        self.tfidf_model = SparseMatrixSimilarity.load(checkpoint_path + "/tfidf")
        self.bm25_index = TfidfModel.load(checkpoint_path + "/bm25_index")
        self.title2idx = title2idx
        self.entity_dict = entity_dict

    def get_topk_stage1(self, query, topk=100):
        tokenized_query = query.split()
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(tokenized_query)]
        scores = self.bm25_index[tfidf_query]
        top_n = np.argsort(scores)[::-1][:topk]
        return top_n, scores[top_n]

    def get_topk_stage2(self, x, raw_answer=None, topk=50):
        x = str(x)
        query = preprocess(x, max_length=128).lower().split()
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query)]
        scores = self.bm25_index[tfidf_query]
        top_n = list(np.argsort(scores)[::-1][:topk])
        if raw_answer is not None:
            raw_answer = raw_answer.strip()
            if raw_answer in self.entity_dict:
                title = self.entity_dict[raw_answer].replace("wiki/", "").replace("_", " ")
                extra_id = self.title2idx.get(title, -1)
                if extra_id != -1 and extra_id not in top_n:
                    top_n.append(extra_id)
        scores = scores[top_n]
        return np.array(top_n), np.array(scores)
