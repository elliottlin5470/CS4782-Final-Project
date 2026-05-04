For this project, you will need to download the SST-2 dataset and RoBERTa-base pretrained transformer model. The code to do so is in our Jupyter notebook:

```python
model_id = "FacebookAI/roberta-base"
dataset_id = "stanfordnlp/sst2"

sst2_dataset = load_dataset(dataset_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
roberta_pretrained = AutoModel.from_pretrained(model_id)
```
