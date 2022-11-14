import os
import json
l = []

for fn in os.listdir('.'):
    fn: str
    if fn.startswith('P') and fn.endswith('.json'):
        with open(fn, 'r') as f:
            c = f.read()
            for cl in c.split('\n'):
                cl = cl.strip()
                if cl:
                    l.append(cl)

l.sort(key=lambda x: json.loads(x)['dimension'])
with open('M1.json', 'w') as f:
    f.write('\n'.join(l))
