import json
import re
from difflib import SequenceMatcher
from utils import *
from valid import *
import time
import pandas as pd
from agents import *
from evaluation import calculate_metrics
from propaganda import apply_propaganda_technique
from detector import model, bert_predict_with_chunking, explain_fake_text, tokenizer, FakeNewsDetector
import os
import csv
import argparse

parser = argparse.ArgumentParser(description="Fake News Generation Pipeline")
parser.add_argument("--mode", type=int, choices=[1, 2, 3], help="1 = UniversalAgent, 2 = all agents active, 3 = deactivate one agent")
parser.add_argument("--disable", type=int, choices=[1, 2, 3, 4], help="Agent to disable (if mode=3): 1=Semantic, 2=Salient, 3=Narrative, 4=Number")
parser.add_argument("--rounds", type=int, help="Number of modification rounds (must be >= 1)")
parser.add_argument("--detector", type=str, choices=["bert", "llm"], default="bert", help="Method to use for fake news detection (bert or llm)")
args = parser.parse_args()

newsdetector = FakeNewsDetector(
    method=args.detector,
    model=model,
    tokenizer=tokenizer,
    llm_agent=LLMFakenewsAgent
)
# === Fallback via input() se mancano o non validi ===
agent_numbers = {
    1: "SemanticAnalyzer",
    2: "SalientSentenceEditor",
    3: "NarrativeModifier",
    4: "NumberModifier"
}

# MODE
while args.mode not in [1, 2, 3]:
    print("Select execution mode:")
    print("1 = UniversalAgent (single all-in-one agent)")
    print("2 = All agents active")
    print("3 = Deactivate one agent")
    try:
        args.mode = int(input("Enter 1, 2 or 3: ").strip())
    except ValueError:
        pass

# DISABLE (only if mode 3)
if args.mode == 3:
    while args.disable not in [1, 2, 3, 4]:
        print("Which agent do you want to deactivate?")
        for number, name in agent_numbers.items():
            print(f"{number} = {name}")
        try:
            args.disable = int(input("Enter number 1–4: ").strip())
        except ValueError:
            pass

# ROUNDS
while args.rounds is None or args.rounds < 1:
    try:
        args.rounds = int(input("How many modification rounds? (min 1): ").strip())
    except ValueError:
        pass

# === Final configuration ===
USE_FULL_AGENT = args.mode == 1
MANUALLY_DISABLED = []

if args.mode == 3:
    MANUALLY_DISABLED = [agent_numbers[args.disable]]

max_rounds = args.rounds

DISABLED_AGENTS = set(MANUALLY_DISABLED)
if "SalientSentenceEditor" in DISABLED_AGENTS:
    DISABLED_AGENTS.add("NarrativeModifier_Feedback")
if "NumberModifier" in DISABLED_AGENTS:
    DISABLED_AGENTS.add("NarrativeModifier_Numbers")


if USE_FULL_AGENT:
    output_csv = "metrics/full_metric.csv"
    output_file = "output_files/output_full.csv"
else:
    if MANUALLY_DISABLED:
        agent_code = MANUALLY_DISABLED[0]
        output_csv = f"metrics/metric_no_{agent_code}.csv"
        output_file = f"output_files/output_no_{agent_code}.csv"
    else:
        output_csv = "metrics/metric.csv"
        output_file = "output_files/output.csv"

os.makedirs("metrics", exist_ok=True)
if not os.path.exists(output_csv):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "index",
            "initial_bleu", "initial_rouge1", "initial_rouge2", "initial_rougeL", "initial_readability", "initial_bert_score",
            "final_bleu", "final_rouge1", "final_rouge2", "final_rougeL", "final_readability", "final_bert_score"
        ])

os.makedirs("output_files", exist_ok=True)
if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["original_title", "original_text", "modified_text_1", "modified_title", "modified_text", "initial_label_score", "final_label_score", "error"])
        writer.writeheader()


log_file = "agent_logs/agent_responses.csv"
os.makedirs("agent_logs", exist_ok=True)
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "agent_name", "input_message", "response_text", "round", "shap_explanation"])
        writer.writeheader()


#funzione per misurare il tempo di esecuzione di ogni agente
def measure_agent_time(agent_name, function, *args, **kwargs):
    start_time = time.time()
    response = function(*args, **kwargs)  
    end_time = time.time()
    execution_time = end_time - start_time
    return response, execution_time

#inizializza il tempo per gli agenti
agent_metrics = {}
ALL_AGENTS = [
    "SemanticAnalyzer", "SalientSentenceEditor", "NarrativeModifier", "NumberModifier",
    "TitleEditor", "Detector", "EvaluatorAgent", "SalientTextRewriter", "UniversalAgent"
]

for agent in ALL_AGENTS:
    agent_metrics[agent] = 0.0

start_time_total = time.time()  

file_path = "file/true1.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Il file {file_path} non esiste. Assicurati che sia nella cartella 'file/'.")

df = pd.read_csv(file_path, delimiter=';')
df = df[['title', 'text']].dropna()

for i in range(3100, 20000):  
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
                if USE_FULL_AGENT:
                    #newspaper_name = "BBC News, Reuters, The Guardian, The New York Times"
                    #newspaper_url = "https://www.bbc.com, https://www.reuters.com, https://www.theguardian.com, https://www.nytimes.com"
                    with open("prompt/fullagent", "r") as f:
                        selection_prompt = f.read()
                    full_message = f"Instructions:\n{selection_prompt}\nText:\n{original_text}"
                    #full_message = f"Instructions:\n{selection_prompt}\nFollow the journalistic style of {newspaper_name} {newspaper_url}\nText:\n{original_text}"
                    response, exec_time = measure_agent_time("UniversalAgent", user_proxy.initiate_chat, UniversalAgent, message=full_message)
                    agent_metrics["UniversalAgent"] = exec_time
                    parsed = valid_modifiedtext(response.summary, UniversalAgent, original_text)
                    modified_text = parsed.get("modified_text", original_text)
                    #text = modified_text
                else:
                    text = original_text 

            else:
                if USE_FULL_AGENT:
                    message = f"Apply this feedback to improve the article:\n{feedback}\nOriginal text:\n{modified_text_final}"
                    response, exec_time = measure_agent_time("UniversalAgent", user_proxy.initiate_chat, UniversalAgent,message=message)
                    agent_metrics["UniversalAgent"] += exec_time
                    parsed = valid_modifiedtext(response.summary, UniversalAgent, modified_text_final)
                    modified_text = parsed.get("modified_text", modified_text_final)
                else:
                    message = f"Apply this feedback:{feedback}\n to the text:\n:{modified_text_final}"
                    response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=message)
                    agent_metrics["NarrativeModifier"] += exec_time
                    parsed_r = valid_modifiedtext(response.summary, NarrativeModifier, original_text=modified_text_final)
                    modified_text = parsed_r.get("modified_text", "")
                    text = modified_text
                    response_to_log = response.summary if isinstance(parsed_r, str) else json.dumps(parsed_r)
                    log_agent_response(i, "NarrativeModifier", message, response_to_log, round_count)

            if not USE_FULL_AGENT:
                sentence_to_modify = ""
                modified_sentence = ""
                numbers = []

                if "SemanticAnalyzer" not in DISABLED_AGENTS:
                    message = f"Analyze the text:\n{text}"
                    response, exec_time = measure_agent_time("SemanticAnalyzer", user_proxy.initiate_chat, SemanticAnalyzer, message=message)
                    agent_metrics["SemanticAnalyzer"] = exec_time
                    analysis_data = valid_semantic(response.summary, SemanticAnalyzer, text)
                    response_to_log = response.summary if isinstance(analysis_data, str) else json.dumps(analysis_data)
                    log_agent_response(i, "SemanticAnalyzer", message, response_to_log, round_count)
                    key_sentences = analysis_data.get("key_sentences", [])
                    numbers = analysis_data.get("numbers", [])
                    sentence_to_modify = key_sentences[0] if key_sentences else text.split(".")[0] + "."
                else:
                    sentence_to_modify = text.split(".")[0] + "." if "." in text else text
                    numbers = []
                
                if "SalientSentenceEditor" not in DISABLED_AGENTS:    
                    message = f"Modify this sentence: {sentence_to_modify}"
                    response, exec_time = measure_agent_time("SalientSentenceEditor", user_proxy.initiate_chat, SalientSentenceEditor, message=message)
                    agent_metrics["SalientSentenceEditor"] = exec_time
                    parsed_response = valid_sentence(response.summary, SalientSentenceEditor, sentence_to_modify)
                    modified_sentence = parsed_response.get("modified_sentence","")
                    response_to_log = response.summary if isinstance(parsed_response, str) else json.dumps(parsed_response)
                    log_agent_response(i, "SalientSentenceEditor", message, response_to_log, round_count)
                else:
                    modified_sentence = sentence_to_modify

                if "NarrativeModifier_Feedback" not in DISABLED_AGENTS:
                    with open("prompt/propaganda_feedback", "r") as f:
                        propaganda_feedback_prompt = f.read()
            
                    message = f"{propaganda_feedback_prompt}\nSentence: {modified_sentence}"
                    response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=message)
                    agent_metrics["NarrativeModifier"] = exec_time
                    parsed_data = valid_feedback(response.summary, NarrativeModifier, default_feedback="Please revise the sentence")
                    feedback = parsed_data.get("feedback", "")
                    response_to_log = response.summary if isinstance(parsed_data, str) else json.dumps(parsed_data)
                    log_agent_response(i, "NarrativeModifier", message, response_to_log, round_count)

                    if feedback and "SalientSentenceEditor" not in DISABLED_AGENTS:
                        message = f"Revise the sentence based on this feedback: {feedback}"
                        response, exec_time = measure_agent_time("SalientSentenceEditor", user_proxy.initiate_chat, SalientSentenceEditor, message=message)
                        agent_metrics["SalientSentenceEditor"] += exec_time
                        parsed_d = valid_sentence(response.summary, SalientSentenceEditor, sentence_to_modify)
                        modified_sentence = parsed_d.get("modified_sentence", "")
                        response_to_log = response.summary if isinstance(parsed_d, str) else json.dumps(parsed_d)
                        log_agent_response(i, "SalientSentenceEditor", message, response_to_log, round_count)
                    else:
                        pass

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
                parsed_ = valid_modifiedtext(response.summary, SalientTextRewriter, original_text=text)
                text = parsed_.get("modified_text", "")
                response_to_log = response.summary if isinstance(parsed_, str) else json.dumps(parsed_)
                log_agent_response(i, "SalientTextRewriter", message, response_to_log, round_count)

                if "NarrativeModifier" not in DISABLED_AGENTS:
                    modified_text = apply_propaganda_technique(text, i, round_count)
                else:
                    modified_text = text

                if "NumberModifier" not in DISABLED_AGENTS:    
                    with open("prompt/number_feedback", "r") as f:
                        number_feedback_prompt = f.read()

                    message = f"Modify numbers in text:\n{modified_text}"
                    response, exec_time = measure_agent_time("NumberModifier", user_proxy.initiate_chat, NumberModifier, message=message)
                    agent_metrics["NumberModifier"] = exec_time
                    parsed_number = valid_modifiedtext(response.summary, NumberModifier, original_text=modified_text)
                    modified_text = parsed_number.get("modified_text", "")
                    response_to_log = response.summary if isinstance(parsed_number, str) else json.dumps(parsed_number)
                    log_agent_response(i, "NumberModifier", message, response_to_log, round_count)

                if "NarrativeModifier_Numbers" not in DISABLED_AGENTS:
                    message = f"{number_feedback_prompt}\nText: {modified_text}"
                    response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=message)
                    agent_metrics["NarrativeModifier"] += exec_time
                    parsed_feed = valid_feedback(response.summary, NarrativeModifier, default_feedback="Please revise number consistency.")
                    feedback = parsed_feed.get("feedback", "")
                    response_to_log = response.summary if isinstance(parsed_feed, str) else json.dumps(parsed_feed)
                    log_agent_response(i, "NarrativeModifier", message, response_to_log, round_count)

                    if feedback:
                        message = f"Revise numbers based on feedback:\n{feedback}\nOriginal text:\n{modified_text}"
                        response, exec_time = measure_agent_time("NumberModifier", user_proxy.initiate_chat, NumberModifier, message=message)
                        agent_metrics["NumberModifier"] += exec_time
                        parsed_n = valid_modifiedtext(response.summary, NumberModifier, original_text=modified_text)
                        modified_text = parsed_n.get("modified_text", "")
                        response_to_log = response.summary if isinstance(parsed_n, str) else json.dumps(parsed_n)
                        log_agent_response(i, "NumberModifier", message, response_to_log, round_count)
                    else:
                        pass
                        
            def evaluate_text_with_agent(original_text, modified_text, index):       
                evaluation_metrics = calculate_metrics(original_text, modified_text)

                if USE_FULL_AGENT:
                    target_agents = ["UniversalAgent"]
                else:
                    target_agents = ["NarrativeModifier", "NumberModifier"]

                message_content = json.dumps({
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "evaluation_metrics": evaluation_metrics,
                    "target_agents": target_agents
                })

                evaluation_response, exec_time = measure_agent_time(
                    "EvaluatorAgent", 
                    user_proxy.initiate_chat, 
                    EvaluatorAgent, 
                    message=message_content
                )

                agent_metrics["EvaluatorAgent"] = exec_time
                
                evaluation_data = valid_evaluator(evaluation_response.summary)
                feedback_data = evaluation_data.get("feedback", [])

                evaluation_results = {
                    "bleu": evaluation_metrics["bleu"],
                    "rouge": evaluation_metrics["rouge"],
                    "readability": evaluation_metrics["readability"],
                    "bert_score": evaluation_metrics["bert_score"],
                    "feedback": feedback_data
                }

                log_agent_response(index, "EvaluatorAgent", message_content, evaluation_response.summary, round_count)

                if USE_FULL_AGENT and feedback_data:
                    for feedback in feedback_data:
                        agent_name = feedback.get("agent", "Unknown")
                        message = feedback.get("message", "")
                        agent_message = f"""{message}
                        Ensure the revised text completely replaces the original passage.
                        text: "{modified_text}"
                        """
                        response, exec_time = measure_agent_time("UniversalAgent", user_proxy.initiate_chat, UniversalAgent, message=agent_message)
                        agent_metrics["UniversalAgent"] += exec_time
                        parsed = valid_modifiedtext(response.summary, UniversalAgent, modified_text)
                        modified_text = parsed.get("modified_text", modified_text)

                # === PIPELINE CLASSICA ===
                elif not USE_FULL_AGENT and feedback_data:
                    for feedback in feedback_data:
                        agent_name = feedback.get("agent", "Unknown")
                        message = feedback.get("message", "")

                        if agent_name == "NarrativeModifier":
                            narrative_message = f"""{message}
                            Ensure the revised text completely replaces the original passage.
                            text: "{modified_text}"
                            """
                            response, exec_time = measure_agent_time("NarrativeModifier", user_proxy.initiate_chat, NarrativeModifier, message=narrative_message)
                            agent_metrics["NarrativeModifier"] += exec_time
                            parsed_json = valid_modifiedtext(response.summary, NarrativeModifier, original_text=modified_text)
                            modified_text = parsed_json.get("modified_text","")
                            response_to_log = response.summary if isinstance(parsed_json, str) else json.dumps(parsed_json)
                            log_agent_response(index, "NarrativeModifier", narrative_message, response_to_log, round_count)

                        if agent_name == "NumberModifier":
                            number_message = f"""{message}
                            Ensure all number-related corrections are applied properly.
                            text: "{modified_text}"
                            """
                            response, exec_time = measure_agent_time("NumberModifier", user_proxy.initiate_chat, NumberModifier, message=number_message)
                            agent_metrics["NumberModifier"] += exec_time
                            parsed_j = valid_modifiedtext(response.summary, NumberModifier, original_text=modified_text)
                            modified_text = parsed_j.get("modified_text","")
                            response_to_log = response.summary if isinstance(parsed_j, str) else json.dumps(parsed_j)
                            log_agent_response(index, "NumberModifier", number_message, response_to_log, round_count)
                        
                    if not evaluation_results["feedback"]:
                        pass

                return modified_text, evaluation_results
            
            evaluation_start_time = time.time()
            modified_text_final, evaluation_results = evaluate_text_with_agent(original_text, modified_text, i)
            evaluation_end_time = time.time()
            final_metrics = calculate_metrics(original_text, modified_text_final)
            evaluation_results.update(final_metrics)

            agent_metrics["EvaluatorAgent_Total"] = evaluation_end_time - evaluation_start_time

            label, probs = newsdetector.predict(modified_text_final)
            real_score = round(float(probs[0]), 4)
            fake_score = round(float(probs[1]), 4)
            label_score_str = f"{label} (Fake: {fake_score}, Real: {real_score})"
            
            if round_count == 0:
                modified_text_1 = modified_text_final
                initial_label_score = label_score_str
                initial_metrics = calculate_metrics(original_text, modified_text_final)

            if label == "Real":
                final_label_score = label_score_str
                final_metrics = calculate_metrics(original_text, modified_text_final)
                modified_text = modified_text_final
                break
            
            if label == "Fake":
                important_words, key_phrases = explain_fake_text(modified_text_final)
                message=f"Revise the text based on this words and phrases:\n{important_words}\n{key_phrases}\nText: {modified_text_final}"
                response, exec_time = measure_agent_time("Detector", user_proxy.initiate_chat, Detector, message=message)
                parsed_fb = valid_feedback(response.summary, Detector, default_feedback="Please revise the text based on highlighted weaknesses." )
                feedback = parsed_fb.get("feedback", "")
                response_to_log = response.summary if isinstance(parsed_fb, str) else json.dumps(parsed_fb)
                log_agent_response(i, "Detector", message, response_to_log, round_count, shap_words=important_words, shap_phrases=key_phrases)
            round_count += 1 

        final_label_score = label_score_str
        final_metrics = calculate_metrics(original_text, modified_text_final)
        modified_text = modified_text_final
        
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

        message = f"Generate a new title based on this text:\n{modified_text_final}"
        response, exec_time = measure_agent_time("TitleEditor", user_proxy.initiate_chat, TitleEditor, message=message)
        agent_metrics["TitleEditor"] = exec_time
        parsed_title = valid_title(response.summary, original_title=title)
        modified_title = parsed_title.get("title", title)
        response_to_log = response.summary if isinstance(parsed_title, str) else json.dumps(parsed_title)
        log_agent_response(i, "TitleEditor", message, response_to_log, round_count)

        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["original_title", "original_text", "modified_text_1", "modified_title", "modified_text", "initial_label_score", "final_label_score", "error"])
            writer.writerow({
                "original_title": original_title,
                "original_text": original_text,
                "modified_text_1": modified_text_1,
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
        print(f"[ERROR] Error processing article {i}: {e}")        

        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["original_title", "original_text", "modified_title", "modified_text", "initial_label_score", "final_label_score", "error"])
            writer.writerow({
                "original_title": title if 'title' in locals() else "",
                "original_text": text if 'text' in locals() else "",
                "modified_text_1": "",
                "modified_title": "",
                "modified_text": "",
                "initial_label_score": "",
                "final_label_score": "",
                "error": str(e)
            })

        continue

    