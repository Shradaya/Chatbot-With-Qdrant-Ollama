from .utils import dotdict
from rouge_score import rouge_scorer


def get_rouge_score(context: str, generated_answer: str, perl_results: list = ['rouge1']):
    scorer = rouge_scorer.RougeScorer(perl_results, use_stemmer=True)
    
    scores = scorer.score(context, generated_answer)
    
    return dotdict(scores)
