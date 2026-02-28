import json
from collections import Counter

layers = Counter()
with open('data/training_pairs.jsonl') as f:
    for line in f:
        p = json.loads(line)
        layer = p.get('layer', p.get('type', 'unknown'))
        instr = p.get('instruction', '')

        if 'OWASP vulnerability categories are statistically likely' in instr:
            key = 'owasp_cooc_t1'
        elif 'suggest' in instr and 'also be present' in instr:
            key = 'owasp_cooc_t2'
        elif "CWE family" in instr or "weakness family" in instr:
            key = 'cwe_family'
        elif 'CISA KEV' in instr or ('actively exploited' in instr and 'co-exploited' in instr):
            key = 'kev_corr_t1'
        elif 'actively exploited' in instr and 'priority of testing' in instr:
            key = 'kev_corr_t2'
        elif 'blog analysis' in instr:
            key = 'blog_cooc_page'
        elif 'exploit chains' in instr and 'chained' in instr:
            key = 'blog_cooc_chain'
        elif 'attack campaigns' in instr:
            key = 'blog_cooc_camp'
        elif layer in ('vulnerability_cooccurrence', 'positive_inference', 'negative_inference', 'chain_reasoning', 'stack_profile', 'conditional_reasoning'):
            key = f'cooc_gen_{p.get("type", layer)}'
        elif 'correlated' in instr or 'co-occur' in instr:
            key = 'corr_graph_t1'
        elif 'attack surface' in instr and 'test first' in instr:
            key = 'corr_graph_t2'
        elif 'threat intelligence perspective' in instr:
            key = 'corr_graph_t3'
        elif 'ATT&CK techniques' in instr:
            key = 'corr_graph_t4'
        elif 'Identify CVEs that form exploit' in instr:
            key = 'corr_graph_t5'
        else:
            key = layer

        layers[key] += 1

print("Training pair breakdown by sub-generator:\n")
for k, c in layers.most_common(30):
    print(f'  {k:45s} {c:>8,}')
print(f'\n  {"TOTAL":45s} {sum(layers.values()):>8,}')

# Aggregate into groups
corr = sum(c for k,c in layers.items() if k.startswith('corr_graph'))
owasp = sum(c for k,c in layers.items() if k.startswith('owasp_cooc'))
cwe = layers.get('cwe_family', 0)
kev = sum(c for k,c in layers.items() if k.startswith('kev_corr'))
blog = sum(c for k,c in layers.items() if k.startswith('blog_cooc'))
cooc = sum(c for k,c in layers.items() if k.startswith('cooc_gen'))
other = sum(layers.values()) - corr - owasp - cwe - kev - blog - cooc

total = sum(layers.values())
print(f'\nGrouped:')
print(f'  Correlation graph pairs:     {corr:>8,}  ({corr/total*100:.1f}%)')
print(f'  OWASP co-occurrence pairs:   {owasp:>8,}  ({owasp/total*100:.1f}%)')
print(f'  CWE family pairs:            {cwe:>8,}  ({cwe/total*100:.1f}%)')
print(f'  KEV correlation pairs:       {kev:>8,}  ({kev/total*100:.1f}%)')
print(f'  Blog co-occurrence pairs:    {blog:>8,}  ({blog/total*100:.1f}%)')
print(f'  Co-occurrence gen pairs:     {cooc:>8,}  ({cooc/total*100:.1f}%)')
print(f'  Other layers:                {other:>8,}  ({other/total*100:.1f}%)')
