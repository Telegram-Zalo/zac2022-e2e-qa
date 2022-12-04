import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, pipeline
from text_utils import post_process_answer
from graph_utils import find_best_cluster


class QAEnsembleModel(nn.Module):

    def __init__(self, model_name, model_checkpoints, entity_dict,
                 thr=0.1, device="cuda:0"):
        super(QAEnsembleModel, self).__init__()
        self.nlps = []
        for model_checkpoint in model_checkpoints:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).half()
            model.load_state_dict(torch.load(model_checkpoint))
            nlp = pipeline('question-answering', model=model,
                           tokenizer=model_name, device=int(device.split(":")[-1]))
            self.nlps.append(nlp)
        self.entity_dict = entity_dict
        self.thr = thr

    def forward(self, question, texts, ranking_scores=None):
        if ranking_scores is None:
            ranking_scores = np.ones((len(texts),))

        curr_answers = []
        curr_scores = []
        best_score = 0
        for i, nlp in enumerate(self.nlps):
            for text, score in zip(texts, ranking_scores):
                QA_input = {
                    'question': question,
                    'context': text
                }
                res = nlp(QA_input)
                # print(res)
                if res["score"] > self.thr:
                    curr_answers.append(res["answer"])
                    curr_scores.append(res["score"])
                res["score"] = res["score"] * score
                if i == 0:
                    if res["score"] > best_score:
                        answer = res["answer"]
                        best_score = res["score"]
        if len(curr_answers) == 0:
            return None
        curr_answers = [post_process_answer(x, self.entity_dict) for x in curr_answers]
        answer = post_process_answer(answer, self.entity_dict)
        new_best_answer = post_process_answer(find_best_cluster(curr_answers, answer), self.entity_dict)
        return new_best_answer
