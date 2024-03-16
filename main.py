from rouge_score import rouge_scorer
from nltk.translate import meteor_score
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
import json

if __name__ == "__main__":

    # Percorso al file JSON
    file_path = 'cases/grizzly.json'

    # Apertura e caricamento del file JSON in un oggetto Python
    with open(file_path, 'r') as file:
        case = json.load(file)

    # Ora l'oggetto `case` contiene i dati del file JSON
    print(case)

    ref = str(case['human'])
    gen = str(case['generated'])
    ben = str(case['benchmark'])
    bad = str(case['bad'])

    print(f"human done:\n{ref}")
    print("\n")
    print(f"generated:\n{gen}")
    print("\n")
    print(f"benchmark:\n{ben}")
    print("\n")
    print(f"bad:\n{bad}")

    print(f"------- ROUGE 3 ---------")
    rscorer = rouge_scorer.RougeScorer(["rouge3"], use_stemmer=True)
    sgener = rscorer.score ( ref , gen )['rouge3'][2]
    sbench  = rscorer.score( ref , ben )['rouge3'][2]
    sbad   = rscorer.score ( ref , bad )['rouge3'][2]



    print(f"gen:{sgener}")
    print(f"ben:{sbench}")
    print(f"bad:{sbad}")


    print(f"------- ROUGE L ---------")
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sgener = rscorer.score( ref , gen )['rougeL'][2]
    sbench  = rscorer.score( ref , ben )['rougeL'][2]
    sbad   = rscorer.score ( ref , bad )['rougeL'][2]            


    print(f"gen:{sgener}")
    print(f"ben:{sbench}")
    print(f"bad:{sbad}")
    

    print(f"------- BLEU ---------")
    reference = [tuple(ref.split())]
    candidate = [tuple(gen.split())]
    benchmark = [tuple(ben.split())]
    badsplit       = [tuple(bad.split())]

    sgener = sentence_bleu(reference, candidate)
    sbench = sentence_bleu(reference, benchmark)
    sbad   = sentence_bleu(reference, badsplit)

    print(f"gen:{sgener}")
    print(f"ben:{sbench}")
    print(f"bad:{sbad}")

    print(f"------- METEOR ---------")
    tokenized_reference = word_tokenize(ref)
    tokenized_candidate = word_tokenize(gen)
    tokenized_benchmark = word_tokenize(ben)
    tokenized_bad = word_tokenize(bad)

    sgener = meteor_score.meteor_score([tokenized_reference], tokenized_candidate)
    sbench = meteor_score.meteor_score([tokenized_reference], tokenized_benchmark)
    sbad   = meteor_score.meteor_score([tokenized_reference], tokenized_bad)
    print(f"gen:{sgener}")
    print(f"ben:{sbench}")
    print(f"bad:{sbad}")

    print(f"------- BERT ---------")
    Pgen, Rgen, F1gen = score([gen], [ref], lang='en', verbose=False)
    Pben, Rben, F1ben = score([ben], [ref], lang='en', verbose=False)
    Pbad, Rbad, F1bad = score([bad], [ref], lang='en', verbose=False)

    sgener, sbench, sbad = F1gen, F1ben, F1bad

    
    print(f"gen:{sgener}")
    print(f"ben:{sbench}")
    print(f"bad:{sbad}")