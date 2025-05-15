import json
import os
import re
import numpy as np
import pandas as pd

"""
llama3 = {
    "config_list": [
        {
            "model": "SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  
}
"""

llama3 = {
    "config_list" : [
        {
            "model": "llama3",  
            "base_url": "http://localhost:11434/v1", 
            "api_key": "ollama",  
        }
    ]
}


