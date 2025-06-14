1. Clone the repository

2. Add files manually
   2.1 Insert the file/ folder (provided in .zip format)
   Inside there is the file true1.csv

project/
└── file/
    └── true1.csv


   2.2 Insert the file bertfakenews1_.pt (already trained BERT model) in the detector/ folder already present in the project.

project/
└── detector/
    └── bertfakenews1_.pt


3. LLaMA 3 Configuration via Ollama

The project uses a LLaMA 3 model served through Ollama.

3.1 Requirements

    Make sure Ollama is installed on the target machine: https://ollama.com

    The "llama3" model must be available in the Ollama instance.

3.2 Editing the utils.py file

In the utils.py file, there are two configuration blocks:

    The first block is commented out and was previously used for an LM Studio setup. It should be ignored.

    THE SECOND BLOCK is active and is the one used by the project. It must be modified to correctly connect to the Ollama instance.

Replace localhost with the IP address or hostname of the machine where Ollama is running, but always keep the /v1 at the end of the URL.

	llama3 = {
		"config_list" : [
		{
			"model": "llama3",
			"base_url": "http://localhost:11434/v1", # <-- CHANGE THIS LINE
			"api_key": "ollama",
		}
	]
}

Update the base_url field by inserting the IP address or hostname of the machine running Ollama.

4. Install dependencies

   Install the necessary packages:

   pip install -r requirements.txt

5. Run the project

   python main.py --mode 2 --rounds 2

Available parameters
    --mode: operating mode

        1 = UniversalAgent (single all-in-one agent)
        2 = All active agents
        3 = Disable a specific agent (also requires --disable)

    --disable: (required if mode=3)

        1 = Semantic
        2 = Salient
        3 = Narrative
        4 = Number

    --rounds: number of modification iterations (≥ 1)

    --detector: method for detecting fake news
        BERT (default)
        LLM   