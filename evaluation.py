import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import textstat
from bert_score import score  

def calculate_metrics(original_text, modified_text):
    """Calcola metriche NLP tra il testo originale e il testo modificato."""
    
    #calcolo BLEU score
    bleu_score = sentence_bleu([original_text.split()], modified_text.split())

    # Calcolo ROUGE score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(original_text, modified_text)

    #calcolo Readability score (Flesch-Kincaid)
    readability_score = textstat.flesch_reading_ease(modified_text)

    #calcolo BERTScore (F1-score medio tra le embeddings dei testi)
    P, R, F1 = score([modified_text], [original_text], lang="en")  
    bert_score_f1 = F1.mean().item()  

    return {
        "bleu": bleu_score,
        "rouge": {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
        },
        "readability": readability_score,
        "bert_score": bert_score_f1
    }
