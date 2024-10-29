from .utils import dotdict
from bert_score import BERTScorer

def get_bert_score(context: str, generated_answer: str):
    scorer = BERTScorer(model_type = 'bert-base-uncased')
    
    P, R, F1 = scorer.score([context], [generated_answer])
    
    return dotdict({
                    'precision': P,
                    'recall': R,
                    'fmeasure': F1    
                })