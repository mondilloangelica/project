import json
import re
from difflib import SequenceMatcher
from utils import get_valid_analysis, fix_messy_json, fix_and_parse_json, get_valid_analysis_messy
import time
import pandas as pd
from agents import user_proxy, SemanticAnalyzer, SalientSentenceEditor, NarrativeModifier, NumberModifier, TitleEditor, EvaluatorAgent, SalientTextRewriter, Detector
from evaluation import calculate_metrics
from propaganda import apply_propaganda_technique
from detector import model, bert_predict_with_chunking, explain_fake_text
import os
import csv

output_csv = "metrics/metric.csv"
os.makedirs("metrics", exist_ok=True)
if not os.path.exists(output_csv):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "index",
            "initial_bleu", "initial_rouge1", "initial_rouge2", "initial_rougeL", "initial_readability", "initial_bert_score",
            "final_bleu", "final_rouge1", "final_rouge2", "final_rougeL", "final_readability", "final_bert_score"
        ])

output_file = "output_files/output.csv"
os.makedirs("output_files", exist_ok=True)
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["original_title", "original_text", "modified_title", "modified_text", "initial_label_score", "final_label_score", "error"])
        writer.writeheader()

def is_valid_feedback_list(feedback_data):
    return (
        isinstance(feedback_data, list)
        and all(
            isinstance(f, dict)
            and "agent" in f
            and "message" in f
            for f in feedback_data
        )
    )
    
#funzione per misurare il tempo di esecuzione di ogni agente
def measure_agent_time(agent_name, function, *args, **kwargs):
    start_time = time.time()
    response = function(*args, **kwargs)  
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"[{agent_name}] Tempo: {execution_time:.4f}s")
    
    return response, execution_time

#inizializza il tempo per gli agenti
agent_metrics = {}
start_time_total = time.time()  

def robust_agent_response_parser(response_summary):
    def _normalize_feedback_item(agent, message):
        return {"agent": str(agent).strip(), "message": str(message).strip()}

    # Pulisce eventuali delimitatori Markdown
    cleaned = re.sub(r"```(?:json)?", "", response_summary).strip("` \n")

    # 1. Primo tentativo: parsing diretto
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            feedback = parsed.get("feedback")
            if isinstance(feedback, list):
                if all(isinstance(item, dict) and "agent" in item and "message" in item for item in feedback):
                    return {"feedback": feedback}, None
                elif all(isinstance(item, str) for item in feedback):
                    return {"feedback": [_normalize_feedback_item("Unknown", item) for item in feedback]}, None
            elif isinstance(feedback, dict):
                return {
                    "feedback": [
                        _normalize_feedback_item(agent, msg)
                        for agent, msg in feedback.items()
                    ]
                }, None
            elif isinstance(feedback, str):
                return {
                    "feedback": [_normalize_feedback_item("Unknown", feedback)]
                }, None
    except Exception:
        pass

    # 2. Fallback: trova un blocco JSON nel testo
    match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                feedback = parsed.get("feedback")
                if isinstance(feedback, list):
                    return {"feedback": [_normalize_feedback_item("Unknown", f) if isinstance(f, str) else f for f in feedback]}, "Partial fallback"
                elif isinstance(feedback, dict):
                    return {"feedback": [_normalize_feedback_item(agent, msg) for agent, msg in feedback.items()]}, "Fallback on dict"
        except Exception:
            pass

    # 3. Parsing manuale da markdown
    feedback_list = []
    for line in cleaned.splitlines():
        match = re.match(r'^\s*-\s+\*\*(.*?)\*\*:?:?\s+(.*)', line)
        if match:
            agent, message = match.groups()
            feedback_list.append(_normalize_feedback_item(agent, message))

    if feedback_list:
        return {"feedback": feedback_list}, "Manual fallback parsed"

    # 4. Fallback finale: tutto il testo in un solo messaggio
    return {
        "feedback": [_normalize_feedback_item("Unknown", cleaned)]
    }, "Unstructured fallback"

file_path = "file.csv/true1.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Il file {file_path} non esiste. Assicurati che sia nella cartella 'file/'.")

df = pd.read_csv(file_path, delimiter=';')
df = df[['title', 'text']].dropna()

for i in range(20000):  
    try: 
        article = df.iloc[i]
        title = article["title"]
        text = article["text"]
        original_text = text
        original_title = title

        results = []
        round_count = 0  
        max_rounds = 2
        interation = 0

        while round_count < max_rounds:
            print(f"\nRound {round_count + 1}/{max_rounds}...")

            if round_count == 0:
                #response, exec_time = measure_agent_time("SemanticAnalyzer", user_proxy.initiate_chat, SemanticAnalyzer, message=f"Analyze the text:\n{text}")
                #agent_metrics["SemanticAnalyzer"] = exec_time
                text = original_text 
            elif round_count == 1:
                response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=f"Apply this feedback:{feedback}\n to the text:\n:{modified_text_final}")
                agent_metrics["NarrativeModifier"] += exec_time
                
                raw_response = response.summary.strip()
                parsed_json, error_message = fix_and_parse_json(raw_response)
                if not parsed_json:
                    expected_format = """
                    {
                    "modified_text": "Text with modified"
                    }
                    """
                    error_feedback_message = f"""
                    The generated JSON is invalid. Here is the error message:
                    {error_message}
                    Regenerate the JSON following this format:
                    {expected_format}
                    Original output received:
                    {raw_response}               
                    """
                    parsed_json = get_valid_analysis(user_proxy, NarrativeModifier, initial_message=error_feedback_message, expected_format=expected_format)
                modified_text = parsed_json.get("modified_text", modified_text) 
                text = modified_text

            response, exec_time = measure_agent_time("SemanticAnalyzer", user_proxy.initiate_chat, SemanticAnalyzer, message=f"Analyze the text:\n{text}")
            agent_metrics["SemanticAnalyzer"] = exec_time
            raw_json = response.summary
            parsed_data, error_message = fix_messy_json(raw_json)
            if not parsed_data:
                expected_format = """
                {
                "key_sentences": ["Sentence 1.", "Sentence 2."],
                "numbers": ["number1", "number2", "number3"]
                }
                """
                error_feedback_message = f"""
                The generated JSON is invalid. Here is the error message:
                {error_message}

                Please regenerate the JSON following this format:
                {expected_format}

                Original output received:
                {raw_json}
                """
                analysis_data = get_valid_analysis_messy(user_proxy, SemanticAnalyzer, initial_message=error_feedback_message, expected_format=expected_format)
            else:
                analysis_data = parsed_data

            if not analysis_data:
                analysis_data = {
                    "key_sentences": [],
                    "numbers": []
                }    
            key_sentences = analysis_data.get("key_sentences", [])
            numbers = analysis_data.get("numbers", [])
    
            sentence_to_modify = ""
            modified_sentence = ""
            if key_sentences:
                sentence_to_modify = key_sentences[0]
                try:
                    response, exec_time = measure_agent_time("SalientSentenceEditor", user_proxy.initiate_chat, SalientSentenceEditor, message=f"Modify this sentence: {sentence_to_modify}")
                    agent_metrics["SalientSentenceEditor"] = exec_time
                    parsed_response, error_message = fix_and_parse_json(response.summary)

                    if not parsed_response or "modified_sentence" not in parsed_response:
                        print("[WARNING] ➔ Risposta non valida dal SalientSentenceEditor. Provo a correggerla.")
                        
                        expected_format = """
                        {
                            "modified_sentence": "Your modified sentence here"
                        }
                        """

                        error_feedback_message = f"""
                        The generated JSON is invalid. Here is the error message:
                        {error_message}
                        Regenerate the JSON following this format:
                        {expected_format}
                        Original output received:
                        {response.summary}               
                        """

                        parsed_response = get_valid_analysis(user_proxy, SalientSentenceEditor,initial_message=error_feedback_message,expected_format=expected_format)

                    if parsed_response and "modified_sentence" in parsed_response:
                        modified_sentence = parsed_response["modified_sentence"]
                    else:
                        print("[FALLBACK] ➔ Ancora risposta non valida. Uso la frase originale.")
                        modified_sentence = sentence_to_modify
                except Exception as e:
                    print(f"[ERRORE] Errore imprevisto: {e}")
                    modified_sentence = sentence_to_modify
            else:
                print("[ERRORE] ➔ Nessuna frase chiave trovata. Non posso applicare la modifica.")
                modified_sentence = " "

            with open("prompt/propaganda_feedback", "r") as f:
                propaganda_feedback_prompt = f.read()

            try:    
                response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=f"{propaganda_feedback_prompt}\nSentence: {modified_sentence}")
                agent_metrics["NarrativeModifier"] = exec_time
                parsed_data, error_message = fix_and_parse_json(response.summary)
                if not parsed_data:
                    #print("[ERRORE] La risposta non è un JSON valido.")
                    expected_format = """ 
                    {
                    "feedback": "Write your feedback here clearly, in valid JSON format." 
                    }
                    """
                    error_feedback_message = f"""
                    The generated JSON is invalid. Here is the error message:
                    {error_message}
                    Regenerate the JSON following this format:
                    {expected_format}
                    Original output received:
                    {response.summary}               
                    """
                    parsed_response = get_valid_analysis(user_proxy, NarrativeModifier, initial_message=error_feedback_message, expected_format=expected_format)
                    #print("[INFO] ➔ Ho dovuto rimandare una richiesta al NarrativeModifier.")
                else:
                    parsed_response = parsed_data
                    #print("[INFO] ➔ La risposta del NarrativeModifier era valida al primo tentativo.")

                feedback = parsed_response.get("feedback", "") if parsed_response else ""

                if feedback:
                    response, exec_time = measure_agent_time("SalientSentenceEditor", user_proxy.initiate_chat, SalientSentenceEditor, message=f"Revise the sentence based on this feedback: {feedback}")
                    agent_metrics["SalientSentenceEditor"] += exec_time
                    raw_json = response.summary
                    parsed_data, error_message = fix_and_parse_json(raw_json)

                    if not parsed_data:
                        print(f"[ERRORE] Parsing JSON fallito dal SalientSentenceEditor: {error_message}")

                        expected_format = """
                        {
                        "modified_sentence": "Your modified sentence here" 
                        }
                        """
                        error_feedback_message = f"""
                        The generated JSON is invalid. Here is the error message:
                        {error_message}
                        Regenerate the JSON following this format:
                        {expected_format}
                        Original output received:
                        {response.summary}               
                        """                    
                        response_data = get_valid_analysis(user_proxy, SalientSentenceEditor, initial_message=error_feedback_message, expected_format=expected_format)
                        print("[INFO] ➔ Ho dovuto rimandare una richiesta al SalientSentenceEditor.")
                    else:
                        response_data = parsed_data
                        #print("[INFO] ➔ La risposta del SalientSentenceEditor era valida al primo tentativo.")

                    modified_sentence = response_data.get("modified_sentence", modified_sentence)
                else:
                    print("[INFO] ➔ parsed_response è vuoto")

            except Exception as e:
                print(f"[ERRORE] Errore imprevisto: {e}")
                feedback = ""
        
            message = f"""
            You must replace the following sentence in the text.

            sentence_to_modify: "{sentence_to_modify}"

            modified_sentence: "{modified_sentence}"

            text:"{text}"

            Return ONLY the full updated text in this exact JSON format:
            {{
            "modified_text": "the full updated text here"
            }}
            """
            response, exec_time = measure_agent_time("SalientTextRewriter", user_proxy.initiate_chat, SalientTextRewriter, message=message)
            agent_metrics["SalientTextRewriter"] = exec_time  
            parsed_data, error_message = fix_and_parse_json(response.summary)
            if not parsed_data:
                #print("[WARNING] ➔ La risposta del SalientTextRewriter non era valida. Rimando la richiesta.")
                expected_format = """
                {
                "modified_text": "Text with modified sentence"
                }
                """
                error_feedback_message = f"""
                The generated JSON is invalid. Here is the error message:
                {error_message}
                Regenerate the JSON following this format:
                {expected_format}
                Original output received:
                {response.summary}               
                """
                parsed_response = get_valid_analysis(user_proxy, SalientTextRewriter, initial_message=error_feedback_message, expected_format=expected_format)
            else:
                parsed_response = parsed_data
            
            text = parsed_response.get("modified_text", text)

            #print("Testo DOPO la sostituzione della frase:\n", text)
            #print("Risposta SalientSentenceEditor:", response.summary)  

            modified_text = apply_propaganda_technique(text)  

            with open("prompt/number_feedback", "r") as f:
                number_feedback_prompt = f.read()

            response, exec_time = measure_agent_time("NumberModifier", user_proxy.initiate_chat, NumberModifier, message=f"Modify numbers in text:\n{modified_text}")
            agent_metrics["NumberModifier"] = exec_time

            raw_json = response.summary
            parsed_data, error_message = fix_and_parse_json(raw_json)

            if not parsed_data:
                #print("[WARNING] ➔ La risposta del NumberModifier non era valida. Rimando la richiesta.")
                expected_format = """
                {
                "modified_text": "Text with modified numbers"
                }
                """
                error_feedback_message = f"""
                The generated JSON is invalid. Here is the error message:
                {error_message}
                Regenerate the JSON following this format:
                {expected_format}
                Original output received:
                {raw_json}               
                """
                response_data = get_valid_analysis(user_proxy,NumberModifier, initial_message=error_feedback_message, expected_format=expected_format)
            else:
                response_data = parsed_data
                #print("[INFO] ➔ La risposta del NumberModifier era valida al primo tentativo.")

            # Uso della risposta corretta
            if "modified_text" in response_data:
                modified_text = response_data["modified_text"]
            else:
                print("[ERRORE] ➔ Nessun testo modificato ricevuto dal NumberModifier.")
            
            
            response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=f"{number_feedback_prompt}\nText: {modified_text}")
            agent_metrics["NarrativeModifier"] += exec_time    
            raw_json = response.summary
            parsed_data, error_message = fix_and_parse_json(raw_json)

            if not parsed_data:
                #print("[WARNING] ➔ La risposta del NarrativeModifier non era valida. Rimando la richiesta.")
                expected_format = """
                {
                "feedback": " feedback "
                }
                """
                error_feedback_message = f"""
                The generated JSON is invalid. Here is the error message:
                {error_message}
                Regenerate the JSON following this format:
                {expected_format}
                Original output received:
                {raw_json}               
                """
                parsed_response = get_valid_analysis(user_proxy, NarrativeModifier, initial_message=error_feedback_message, expected_format=expected_format)
            else:
                parsed_response = parsed_data
                #print("[INFO] ➔ La risposta del NarrativeModifier era valida al primo tentativo.")

            feedback = parsed_response.get("feedback", "") if parsed_response else ""

            # Se ho ricevuto un feedback valido
            if feedback:
                response, exec_time = measure_agent_time("NumberModifier", user_proxy.initiate_chat, NumberModifier, message=f"Revise numbers based on feedback:\n{feedback}\nOriginal text:\n{modified_text}")
                agent_metrics["NumberModifier"] += exec_time

                raw_json = response.summary
                parsed_data, error_message = fix_and_parse_json(raw_json)

                if not parsed_data:
                    #print("[WARNING] ➔ La risposta del NumberModifier non era valida. Rimando la richiesta.")
                    expected_format = """
                    {
                    "modified_text": "Text with modified numbers"
                    }
                    """
                    error_feedback_message = f"""
                    The generated JSON is invalid. Here is the error message:
                    {error_message}
                    Regenerate the JSON following this format:
                    {expected_format}
                    Original output received:
                    {raw_json}               
                    """
                    response_data = get_valid_analysis(user_proxy, NumberModifier, initial_message=error_feedback_message,expected_format=expected_format)
                else:
                    response_data = parsed_data
                    #print("[INFO] ➔ La risposta del NumberModifier era valida al primo tentativo.")

                # Applico il risultato
                modified_text = response_data.get("modified_text", modified_text)
            else:
                print("[ERRORE] ➔ Nessun feedback ricevuto dal NarrativeModifier.")


            def evaluate_text_with_agent(original_text, modified_text, index):       
                evaluation_metrics = calculate_metrics(original_text, modified_text)

                message_content = json.dumps({
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "evaluation_metrics": evaluation_metrics
                })

                evaluation_response, exec_time = measure_agent_time(
                    "EvaluatorAgent", 
                    user_proxy.initiate_chat, 
                    EvaluatorAgent, 
                    message=message_content
                )

                agent_metrics["EvaluatorAgent"] = exec_time
                print("raw response", evaluation_response.summary)

                evaluation_data, error_message = robust_agent_response_parser(evaluation_response.summary)
                feedback_data = evaluation_data.get("feedback", [])

                # Validazione semantica del feedback
                if not is_valid_feedback_list(feedback_data):
                    print("[ERRORE] Feedback malformato, richiamo l'agente...")

                    expected_format = '''
                    {
                        "feedback": [
                            { "agent": "NameOfAgent", "message": "Feedback message" }
                        ]
                    }
                    '''

                    error_feedback_message = f"""
                    The JSON you returned is invalid or semantically incorrect.
                    You returned feedback in the wrong format — each item must be a dictionary with 'agent' and 'message' keys.
                    For example, this is correct:

                    {expected_format}

                    Please correct your response.
                    Original output received:
                    {evaluation_response.summary}
                    """

                    evaluation_data = get_valid_analysis(
                        user_proxy,
                        EvaluatorAgent,
                        initial_message=error_feedback_message,
                        expected_format=expected_format
                    )
                    feedback_data = evaluation_data.get("feedback", [])
                    print("[INFO] ➔ Ho ricevuto una nuova risposta dal EvaluatorAgent.")
                else:
                    print("[INFO] ➔ La risposta dell'EvaluatorAgent era valida al primo tentativo.")

                evaluation_results = {
                    "bleu": evaluation_metrics["bleu"],
                    "rouge": evaluation_metrics["rouge"],
                    "readability": evaluation_metrics["readability"],
                    "bert_score": evaluation_metrics["bert_score"],
                    "feedback": feedback_data
                }

                for feedback in feedback_data:
                    agent_name = feedback.get("agent", "Unknown")
                    message = feedback.get("message", "")

                    print(f"\nFeedback per {agent_name}:\n{message}")

                    if agent_name == "NarrativeModifier":
                        narrative_message = f"""{message}
                        Ensure the revised text completely replaces the original passage.
                        text: "{modified_text}"
                        """
                        response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=narrative_message)
                        agent_metrics["NarrativeModifier"] += exec_time
                        parsed_json, error_message = fix_and_parse_json(response.summary)
                        if not parsed_json:
                            expected_format = '{ "modified_text": "..." }'
                            parsed_json = get_valid_analysis(user_proxy, NarrativeModifier, initial_message=message, expected_format=expected_format)
                        modified_text = parsed_json.get("modified_text", modified_text)

                    if agent_name == "NumberModifier":
                        number_message = f"""{message}
                        Ensure all number-related corrections are applied properly.
                        text: "{modified_text}"
                        """
                        response, exec_time = measure_agent_time("NumberModifier", user_proxy.initiate_chat, NumberModifier, message=number_message)
                        agent_metrics["NumberModifier"] += exec_time
                        parsed_json, error_message = fix_and_parse_json(response.summary)
                        if not parsed_json:
                            expected_format = '{ "modified_text": "..." }'
                            parsed_json = get_valid_analysis(user_proxy, NumberModifier, initial_message=message, expected_format=expected_format)
                        modified_text = parsed_json.get("modified_text", modified_text)

                if not evaluation_results["feedback"]:
                    print("\n Il testo è già ottimizzato!")

                print("modified_text", modified_text)
                return modified_text, evaluation_results

            
            evaluation_start_time = time.time()
            modified_text_final, evaluation_results = evaluate_text_with_agent(original_text, modified_text, i)
            evaluation_end_time = time.time()
            final_metrics = calculate_metrics(original_text, modified_text_final)
            evaluation_results.update(final_metrics)
            #print("Final evaluation results:", evaluation_results)

            agent_metrics["EvaluatorAgent_Total"] = evaluation_end_time - evaluation_start_time

            probs = bert_predict_with_chunking(model, [modified_text_final])[0]
            label = "Real" if probs[0] > probs[1] else "Fake"
            real_score = round(float(probs[0]), 4)
            fake_score = round(float(probs[1]), 4)
            label_score_str = f"{label} (Fake: {fake_score}, Real: {real_score})"
            
            if round_count == 0:
                initial_label_score = label_score_str
                initial_metrics = calculate_metrics(original_text, modified_text_final)

            if label == "Real":
                final_label_score = label_score_str
                final_metrics = calculate_metrics(original_text, modified_text_final)
                break
            
            if label == "Fake":
                important_words, key_phrases = explain_fake_text(modified_text_final)
                response, exec_time = measure_agent_time("detector", user_proxy.initiate_chat, Detector, message=f"Revise the text based on this words and phrases:\n{important_words}\n{key_phrases}\nText: {modified_text_final}")
                raw_json = response.summary
                parsed_data, error_message = fix_and_parse_json(raw_json)

                if not parsed_data:
                    #print("[WARNING] ➔ La risposta del TitleEditor non era valida. Rimando la richiesta.")
                    expected_format = """
                    {
                    "feedback": "feedback here" 
                    }
                    """
                    error_feedback_message = f"""
                    The generated JSON is invalid. Here is the error message:
                    {error_message}
                    Regenerate the JSON following this format:
                    {expected_format}
                    Original output received:
                    {raw_json}               
                    """    
                    response_data = get_valid_analysis(user_proxy, Detector, initial_message=error_feedback_message, expected_format=expected_format)
                else:
                    response_data = parsed_data
                    #print("[INFO] ➔ La risposta del TitleEditor era valida al primo tentativo.")

                # Applico il risultato
                feedback = response_data.get("feedback", "") if response_data else ""

            round_count += 1 

        final_label_score = label_score_str
        final_metrics = calculate_metrics(original_text, modified_text_final)
        
        rouge = final_metrics["rouge"]
        with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                i,
                initial_metrics.get("bleu", ""),
                initial_metrics.get("rouge", {}).get("rouge1", ""),
                initial_metrics.get("rouge", {}).get("rouge2", ""),
                initial_metrics.get("rouge", {}).get("rougeL", ""),
                initial_metrics.get("readability", ""),
                initial_metrics.get("bert_score", ""),
                final_metrics.get("bleu", ""),
                final_metrics.get("rouge", {}).get("rouge1", ""),
                final_metrics.get("rouge", {}).get("rouge2", ""),
                final_metrics.get("rouge", {}).get("rougeL", ""),
                final_metrics.get("readability", ""),
                final_metrics.get("bert_score", "")
            ])
        response, exec_time = measure_agent_time("TitleEditor", user_proxy.initiate_chat, TitleEditor, message=f"Generate a new title based on this text:\n{modified_text_final}")
        agent_metrics["TitleEditor"] = exec_time
        raw_json = response.summary
        parsed_data, error_message = fix_and_parse_json(raw_json)

        if not parsed_data:
            #print("[WARNING] ➔ La risposta del TitleEditor non era valida. Rimando la richiesta.")
            expected_format = """
            {
            "title": "New Title Generated" 
            }
            """
            error_feedback_message = f"""
            The generated JSON is invalid. Here is the error message:
            {error_message}
            Regenerate the JSON following this format:
            {expected_format}
            Original output received:
            {raw_json}               
            """    
            response_data = get_valid_analysis(user_proxy, TitleEditor, initial_message=error_feedback_message, expected_format=expected_format)
        else:
            response_data = parsed_data
            #print("[INFO] ➔ La risposta del TitleEditor era valida al primo tentativo.")

        # Applico il risultato
        modified_title = response_data.get("title", title)

        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["original_title", "original_text", "modified_title", "modified_text", "initial_label_score", "final_label_score", "error"])
            writer.writerow({
                "original_title": original_title,
                "original_text": original_text,
                "modified_title": modified_title,
                "modified_text": modified_text,
                "initial_label_score": initial_label_score,
                "final_label_score": final_label_score,
                "error": ""
            })
    
        end_time_total = time.time()
        total_execution_time = end_time_total - start_time_total
        
        agent_timing_file = "times/time.csv"
        header = ["index"] + list(agent_metrics.keys()) + ["total_execution_time"]
        os.makedirs("times", exist_ok=True)
        if not os.path.exists(agent_timing_file):
            with open(agent_timing_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
        with open(agent_timing_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [i] + [agent_metrics.get(agent, 0.0) for agent in header[1:-1]] + [total_execution_time]
            writer.writerow(row)

    except Exception as e:
        print(f"[ERRORE] Errore durante l'elaborazione dell'articolo {i}: {e}")        

        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["original_title", "original_text", "modified_title", "modified_text", "initial_label_score", "final_label_score", "error"])
            writer.writerow({
                "original_title": title if 'title' in locals() else "",
                "original_text": text if 'text' in locals() else "",
                "modified_title": "",
                "modified_text": "",
                "initial_label_score": "",
                "final_label_score": "",
                "error": str(e)
            })

        continue

    