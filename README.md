# RAE
This repository hosts the code for our 'Retrieval-Augmented In-context Model Editing for Multi-hop Question Answering‘

## Data
Please use the following code to retrieve the data:
```
cat xa* > data.zip
```


## Dependencies
Please see requirement.txt

## Running
### Edit on MQUAKE-CF-3k
```
python main.py --model gpt2 --mode beam --dataset MQuAKE-CF-3k
```
### Edit on MQUAKE-T
```
python main.py --model gpt2 --mode beam --dataset MQuAKE-T
```
