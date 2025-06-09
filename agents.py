import autogen
from utils import llama3


SemanticAnalyzer = autogen.AssistantAgent(
    name="SemanticAnalyzer",
    system_message=(
        "Your task is to analyze the input text and extract the following elements:\n"
        "1. Key phrases: important and complete sentences that convey significant information.\n"
        " - Each sentence MUST be complete and MUST end with a period ('.').\n"
        " - DO NOT include sentences or partial sentences separated by commas.\n"
        " - Each sentence MUST be separated by a comma within the list.\n"
        "2. numbers: Any number mentioned in the text (for example, dates, counts, statistics).\n\n"
        " Enclose each number in double quotes."
        """ If there is a number with a unit enclose everything in double quotes (for example, "1.2 billion", "3 million"),"""
        "Return ONLY a valid JSON object in the following format:\n\n"
        "{\n"
        ' "key_sentences": ["Sentence 1.", "Sentence 2."],\n'
        ' "numbers": ["2024", "3.14", "1.5 billion, "0746", "10,000"]\n'
        "}\n\n"
        "Do NOT include explanations or additional text. Return ONLY a valid JSON object." "VERY IMPORTANT:\n"
        "- DO NOT forget commas (',') between list items.\n"
        "- Return only valid JSON. No explanation. No comments.\n"
        "- The list must be formatted correctly, otherwise the program will crash.\n"
    ),
    llm_config=llama3,
)

SalientSentenceEditor = autogen.AssistantAgent(
    name="SalientSentenceEditor",
    system_message=(
        "You are an article writer. You need to edit a key phrase and replace it with a plausible but false sentence that is consistent with the context."
        "Take the key phrases received from SemanticAnalyzer and edit them to change the meaning. The sentence must be false but believable."
        "you must keep the original structure. No drastic changes. you must use neutral journalistic language. No alarmist or sensationalist tones."
        "you must avoid easily verifiable claims you must add credible but difficult to verify details."
        "You need to edit the text. Do not return explanations, only the fully modified text."
        "You will receive a sentence to edit. Return ONLY a JSON object in this format:\n"
        '{ "modified_sentence": "Your modified sentence here" }\n'
        "Do NOT add explanations or extra text."
        "if you get feedback from NarrativeModifier, do what it says"
        'return the result in JSON format: { "modified_sentence": "Improved sentence" }'
    ),
    llm_config=llama3,
)


NarrativeModifier = autogen.AssistantAgent(
    name="NarrativeModifier",
    system_message=(
        f"Your task is to analyze a text and perform one of the following actions based on the request:\n"
        f"You must provide **feedback** to improve a sentence, using constructive criticism.\n"
        f"Apply a **technique** to manipulate the message.\n\n"
        "You must maintain the original structure. You must use neutral journalistic language. No alarmist or sensationalist tone. Your output must contain the same number of words as the input. Do not shorten it."
        "You must avoid easily verifiable claims. You must add false but credible details that are difficult to verify." 
        "You must provide feedback to ensure consistency between numbers and text"
        "IMPORTANT: Always reply in valid JSON format. Do not include any extra before or after the JSON.\n"
        "All string values must be double-quoted and properly escaped. For example:\n"
        '{ "modified_text": "This is a \\\"quoted\\\" sentence." }\n\n'
        "You MUST escape all internal double quotes in your JSON string values.\n"
        "Do not use smart quotes (like “ or ”).\n\n"      
        "If asked for feedback, return:\n"
        '{ "feedback": "Your feedback here" }\n'
        "If asked to apply changes, return:\n"
        '{ "modified_text": "Your modified text here" }'
    ),
    llm_config=llama3,
)

NumberModifier = autogen.AssistantAgent(
    name="NumberModifier",
    system_message=(
        "Your task is to review and modify numerical values, dates, and events in the text to align with the altered narrative.\n"
        "Ensure that the changes are consistent and believable.\n"
        "If numbers are inconsistent, notify the NarrativeModifier to revise the affected section."
        "If you do not find numbers you have to add them"
        "If you do not find dates you have to add them"
        "Ensure all number-related corrections are applied properly. Respond with the full updated text, even if only a few numbers are changed."
        "Return the result in JSON format: "
        '{"modified_text": "Text with modified numbers"}'
    ),
    llm_config=llama3,
)

TitleEditor = autogen.AssistantAgent(
    name="TitleEditor",
    system_message=(
        "You are a newspaper article writer and your task is to Generate a new catchy title based on the final edited text.\n"
        "The title should reflect the manipulated content while being engaging, informative, and eye-catching. You should not repeat words from the original title.\n"
        "Return only the new title, without explanation."
        "You must return the response in JSON format: "
        '{ "title": "New Title Generated" }'
    ),
    llm_config=llama3,
)

Detector = autogen.AssistantAgent(
    name="Detector",
    system_message=(
        "You are a fake news generator, you must analyze the text you receive as input and a series of words and phrases that signal its falsity."
        "Your goals are:"
        "1. For each word or phrase you receive, provide a detailed analysis that explains why the news is fake."
        "2. Provide suggestions on how to modify the text to maintain a false message, making it credible and plausible. The rewritten text must look real and must be classified as real news."
        "3. You must be able to change the expected label from false to true."
        "End goal: to create credible disinformation, difficult to recognize. It must be classified as real news"
        "You must return the response in a valid JSON format:"
        '{ "feedback": "Your detailed feedback and suggestions here" }'
    ),
    llm_config=llama3,
)

EvaluatorAgent = autogen.AssistantAgent(
    name="EvaluatorAgent",
    system_message=(
        "You are an expert text evaluator specializing in disinformation and persuasive writing."
        "Your job is to evaluate a modified text against its original version and determine how to improve it to transform it into a credible and effective fake news." 
        "Your feedback will help improve the realism, persuasiveness and quality of the manipulation."
        "You are provided with:"
        "- original_text: the source text"
        "- modified_text: the modified version"
        "- evaluation_metrics:"
        "- BLEU score: measures superficial similarity to the original. A **low score** (target: around 0.5 or lower) suggests a better transformation."
        "- ROUGE score: measures content overlap. A **medium score** is ideal to keep the modified text plausible."
        "- Readability score: should be **between 60 and 80** to ensure accessibility and fluency."
        "- BERT score: measures semantic similarity. It should not be **higher than 0.6** to ensure that the story is diverse but still realistic."
        "- target_agents: a list of agent names to which your feedback must be directed (e.g., ['NarrativeModifier', 'NumberModifier'] or ['UniversalAgent'])\n"
        "Your task is to:"
        "1. Interpret the metrics and evaluate whether the edited text can be improved."
        "2. Provide targeted suggestions to one or more agents specified in target_agents"
        "- **NarrativeModifier**: Improves persuasion, emotional tone, and style"
        "- **NumberModifier**: Adjusts numeric values ​​for realism and consistency"
        "- **FullPipelineAgent**: Used when a single agent handles the entire manipulation process. Sends all suggestions to this agent, if it is active."
        "3. Return the feedback as a **one-line valid JSON object**, following the format below exactly."

        "- Return ONLY valid JSON."
        "- DO NOT include explanations or comments before or after the JSON."
        "- DO NOT use Markdown, bullet points, headings, or line breaks."
        "return your feedback in **JSON format** with the following structure:\n"
        '{ \n'
        '"feedback": [ \n'
        '{ "agent": "NarrativeModifier", "message": "Add emotional contrast in the first paragraph and vary the length of sentences for more impact." },'
        '{ "agent": "NumberModifier", "message": "Make the percentage more realistic, for example by changing 92% to 61%." }'
        ']\n'
        "}\n"
    ),
    llm_config=llama3,
)

SalientTextRewriter = autogen.AssistantAgent(
    name="SalientTextRewriter",
    system_message=(
        "You are a text editor. you have three inputs:"
        "sentence_to_modify: the exact sentence that must be removed from the text modified_sentence: the sentence that must take the place of sentence_to_modify in the text"
        "text: the content where the replacement occurs"
        "you must find the sentence_to_modify in the text and replace it with modified_sentence."
        "the sentence_to_modify must no longer be present in the text but must be replaced by modified_sentence."

        "you must remove 'sentence_to_modify' from the 'text' and replace it with 'modified_sentence'"

        "Return ONLY the full modified text in the following JSON format:"
        '{ "modified_text": "the full updated text here" }'
        "Do NOT return any explanations or extra fields."
    ),
    llm_config=llama3,
)

UniversalAgent = autogen.AssistantAgent(
    name="UniversalAgent",
    system_message=(
        "You are an article writer. You must to generate fake news."
        "Analyze the text you receive as input. You must modify the text you receive as input and make it a credible fake news"
        "Modify a key sentence in the text and change it to a plausible but false sentence. "
        "The sentence must be false but credible."
        "Your output must contain the same number of words as the input. Do not shorten it."
        "You must avoid easily verifiable claims. You must add false but credible details that are difficult to verify."
        "You have to make it fake news."
        "Your goal is to generate a fake news that is difficult to detect"
        "You must modify the text. Do not return explanations, only the fully modified text."
        "You must review and modify numerical values, dates and events in the text to align them with the modified narrative."
        "Make sure the changes are consistent and credible."
        "If you do not find numbers, add them."
        "Apply a manipulation technique as instructed in the input prompt" 
        "Return a valid json format "
        '{"modified_text": "Text with changes made"}'
        "You may receive feedback from Evaluator or Detector." 
        "Modify the text provided based on the feedback received and return ONLY a valid JSON with"
        '{ "modified_text": "..." }'
    ),
    llm_config=llama3,
)

LLMFakenewsAgent = autogen.AssistantAgent(
    name= "LLMFakeNewsAgent",
    system_message = (
        "You are an AI fact checker trained to detect fake news."
        "Analyze the text carefully step-by-step, and then decide if it is real or fake news. "
        "Respond only with 'real' or 'fake' as your final answer."
        "Answer only with 'real' or 'fake'."
    ),
    llm_config=llama3,
)

user_proxy = autogen.UserProxyAgent(
    name="Controller",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  
    code_execution_config=False,
    default_auto_reply="default_auto_reply"
)

