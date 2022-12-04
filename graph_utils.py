import networkx as nx
import numpy as np
from cdlib import algorithms


# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def is_date_or_num(answer):
    answer = answer.lower().split()
    for w in answer:
        w = w.strip()
        if w.isnumeric() or w in ["ngày", "tháng", "năm"]:
            return True
    return False


def find_best_cluster(answers, best_answer, thr=0.79):
    if len(answers) == 0:  # or best_answer not in answers:
        return best_answer
    elif len(answers) == 1:
        return answers[0]
    dists = np.zeros((len(answers), len(answers)))
    for i in range(len(answers) - 1):
        for j in range(i + 1, len(answers)):
            a1 = answers[i].lower().strip()
            a2 = answers[j].lower().strip()
            if is_date_or_num(a1) or is_date_or_num(a2):
                # print(a1, a2)
                if a1 == a2 or ("tháng" in a1 and a1 in a2) or ("tháng" in a2 and a2 in a1):
                    dists[i, j] = 1
                    dists[j, i] = 1
                # continue
            elif a1 == a2 or (a1 in a2) or (a2 in a1) or compute_f1(a1.lower(), a2.lower()) >= thr:
                dists[i, j] = 1
                dists[j, i] = 1
    # print(dists)
    try:
        thr = 1
        dups = np.where(dists >= thr)
        dup_strs = []
        edges = []
        for i, j in zip(dups[0], dups[1]):
            if i != j:
                edges.append((i, j))
        G = nx.Graph()
        for i, answer in enumerate(answers):
            G.add_node(i, content=answer)
        G.add_edges_from(edges)
        partition = algorithms.louvain(G)
        max_len_comm = np.max([len(x) for x in partition.communities])
        best_comms = []
        for comm in partition.communities:
            # print([answers[i] for i in comm])
            if len(comm) == max_len_comm:
                best_comms.append([answers[i] for i in comm])
        # if len(best_comms) > 1:
        #     return best_answer
        for comm in best_comms:
            if best_answer in comm:
                return best_answer
        mid = len(best_comms[0]) // 2
        # print(mid, sorted(best_comms[0], key = len))
        return sorted(best_comms[0], key=len)[mid]
    except Exception as e:
        print(e, "Disconnected graph")
        return best_answer
