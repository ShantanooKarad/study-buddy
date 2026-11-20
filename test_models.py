import os
import time
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ['GEMINI_API_KEY'],
    http_options=types.HttpOptions(timeout=20000),
)

candidates = [
    'gemini-flash-lite-latest',
    'gemini-3.1-flash-lite-preview',
    'gemini-3-flash-preview',
    'gemini-flash-latest',
    'gemma-4-26b-a4b-it',
]
for name in candidates:
    print('Testing', name, '...', flush=True)
    start = time.time()
    try:
        client.models.generate_content(model=name, contents='Say hi.')
        print('  WORKS', name, round(time.time() - start, 1), 'seconds', flush=True)
    except Exception as error:
        print('  FAILS', name, str(error)[:80], flush=True)