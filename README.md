# Solution for Zalo AI Challenge 2022 - E2E Question Answering

## Overview [WIP]
Pipeline gồm 4 bước chính:

0. Cắt data wikidump thành các sliding windows kích thước 256.
1. Tìm candidate contexts bằng BM25 (Recall@200 ~ 0.95)
2. Rank lại top200 candidate contexts bằng model BERT sentence pair.
3. Tìm candidate answers từ contexts, chọn kết quả cuối cùng bằng mojority vote + community detection w/ Louvain.
4. Tìm top100 candidate articles cho answer bằng BM25, rank lại bằng một model BERT sentence pair khác 
để tìm article cuối cùng.

## Requirements
```
transformers==4.24.0
git+https://github.com/witiko/gensim.git@feature/bm25
```
## Inference example
Tải pretrained models và các data càn thiết từ: [link](https://drive.google.com/file/d/18t2xCvYR4L5vqVO7KpT2lsEmTQDB3r4R/view?usp=share_link), giải nén vào thư mục ```./data/```

Tham khảo notebook ```example```

```
question = "Công ty mẹ của Zalo là gì"
```

Lấy top200 contexts bằng BM25
```
query = preprocess(question).lower()
top_n, bm25_scores = bm25_model_stage1.get_topk_stage1(query, topk=200)
titles = [preprocess(df_wiki_windows.title.values[i]) for i in top_n]
texts = [preprocess(df_wiki_windows.text.values[i]) for i in top_n]
```

Rerank bằng BERT sentence pair
```
question = preprocess(question)
ranking_preds = pairwise_model_stage1.stage1_ranking(question, texts)
ranking_scores = ranking_preds * bm25_scores
```

Tìm câu trả lời tốt nhất bằng model QA 
```
best_idxs = np.argsort(ranking_scores)[-10:]
ranking_scores = np.array(ranking_scores)[best_idxs]
texts = np.array(texts)[best_idxs]
best_answer = qa_model(question, texts, ranking_scores)
```
Entity map để tìm ra câu trả lời cuối cùng
```
bm25_answer = preprocess(str(best_answer).lower(), max_length=128, remove_puncts=True)
bm25_question = preprocess(str(question).lower(), max_length=128, remove_puncts=True)
candidates, scores = bm25_model_stage2_title.get_topk_stage2(bm25_answer, raw_answer=best_answer)
titles = [df_wiki.title.values[i] for i in candidates]
texts = [df_wiki.text.values[i] for i in candidates]
ranking_preds = pairwise_model_stage2.stage2_ranking(question, best_answer, titles, texts)
final_answer = titles[ranking_preds.argmax()]
```
