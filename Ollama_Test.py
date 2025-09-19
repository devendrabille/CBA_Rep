import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

import requests
print(requests.get('http://127.0.0.1:11434/').text)     # "Ollama is running"
print(requests.get('http://127.0.0.1:11434/api/tags').json())
