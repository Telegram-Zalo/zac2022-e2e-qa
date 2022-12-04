Lưu ý: code training đang hơi dơ, trong quá trình refactor lại.
# Training pipeline


# Preprocess 
Lần lượt chạy các notebook sau trong thư mục ```preprocess_scripts```

 1. ```create_sliding_wiki.ipynb```
 2. ```dirtify_data.ipynb```
 3. ```find_redirects.ipynb```
 4. ```create_stage1_ranking.ipynb```
 5. ```create_stage2_ranking.ipynb```
 
Sau khi hoàn tất thư mục ```processed``` có các files sau :
```
processed
│   wiki_20220620_cleaned_v2.csv
│   entities.json
│   train_stage1_ranking.csv
│   train_stage2_ranking.csv
```
# Train các models
## Các model BM25
Chạy các notebook sau:
 1. ```bm25_stage1.ipynb```
 2. ```bm25_stage2.ipynb```
Sau khi hoàn tất thư mục ```outputs``` có các files sau :
```
outputs
│   bm25_stage1
│   bm25_stage2
│   └───title
│   └───full_text
```
## Các model ranking
Chạy các notebook sau:
 1. ```train_pairwise-stage1.ipynb```
 2. ```train_pairwise-stage2.ipynb```

Script thứ 2 sẽ chạy lần để ensemble nếu muốn reproduce kết quả chính xác, lần chạy thứ 2 vui lòng sửa tên file checkpoint ở cuối vd: ```pairwise_stage2_seed1.bin```. 
Sau khi hoàn tất thư mục ```outputs``` có các files sau :
```
outputs
│   pairwise_v2.bin
│   pairwise_stage2_seed0.bin
│   pairwise_stage2_seed1.bin
│   bm25_stage1
│   bm25_stage1
│   bm25_stage2
│   └───title
│   └───full_text
```

## Các model QA
Chạy lại từ các notebook sau trên Kaggle:
|Link  | Checkpoint | Tên tương ứng lúc infer|
|--|--|--|
| [link](https://www.kaggle.com/code/duykhanh99/robust-model-qa-finetune-dataset-add-pseudo/data) | 23915 | qa_model_robust.bin |
| [link](https://www.kaggle.com/code/duykhanh99/new-promax-model-qa-finetune/data) | 24860| qa_model_title.bin |
| [link](https://www.kaggle.com/code/duykhanh99/v2-model-qa-train-all-with-dirty-text-v2/data) | 20283| qa_model_full_20k.bin |
| [link](https://www.kaggle.com/code/duykhanh99/model-qa-train-all-with-dirty-text-v2/data) | 21008| qa_model_full_21k.bin |
