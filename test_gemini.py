import os
from google import genai

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

skip = ('image', 'tts', 'live', 'audio', 'embedding', 'veo', 'imagen')
names = []
for m in client.models.list():
    actions = getattr(m, 'supported_actions', None) or []
    name = m.name.replace('models/', '')
    if 'generateContent' in actions and not any(word in name for word in skip):
        names.append(name)

print('Testing', min(len(names), 12), 'models...')
for name in names[:12]:
    try:
        client.models.generate_content(model=name, contents='Say hi.')
        print('WORKS', name)
    except Exception as error:
        print('FAILS', name, str(error)[:70])

print('Testing embeddings...')
try:
    e = client.models.embed_content(model='gemini-embedding-001', contents='hello')
    print('Embeddings OK')
except Exception as error:
    print('Embeddings FAILED', str(error)[:70])