import json, os

files = {
    'raw_nvd.json': 'NVD CVEs',
    'raw_blogs.json': 'Blog records',
    'raw_correlations.json': 'Correlation records',
    'raw_cooccurrence_v2.json': 'Co-occurrence v2 pairs',
    'raw_cooccurrence.json': 'Co-occurrence v1',
    'raw_cisa_kev.json': 'CISA KEV entries',
    'raw_epss.json': 'EPSS scores',
    'raw_exploitdb.json': 'ExploitDB entries',
    'raw_github.json': 'GitHub advisories',
    'raw_mitre_attack.json': 'MITRE ATT&CK',
    'raw_papers.json': 'Research papers',
    'raw_vendor_advisories.json': 'Vendor advisories',
    'raw_closed.json': 'Closed sources',
    'raw_cwe_chains.json': 'CWE chains',
    'raw_kev_clusters.json': 'KEV clusters',
    'training_pairs.jsonl': 'Training pairs',
    'vuln_dataset.jsonl': 'Vuln dataset',
}

header = f"{'File':<30} {'Records':>10} {'Size MB':>10}"
print(header)
print('-' * 52)
for f, label in files.items():
    path = f'data/{f}'
    if not os.path.exists(path):
        continue
    sz = os.path.getsize(path) / 1024 / 1024
    try:
        if f.endswith('.jsonl'):
            with open(path) as fh:
                count = sum(1 for _ in fh)
        else:
            with open(path) as fh:
                d = json.load(fh)
            if isinstance(d, list):
                count = len(d)
            elif isinstance(d, dict):
                for k in ['cooccurrence_pairs', 'pairs', 'data', 'results']:
                    if k in d:
                        count = len(d[k])
                        break
                else:
                    count = len(d)
            else:
                count = '?'
    except Exception as e:
        count = f'ERR: {e}'
    print(f"{f:<30} {str(count):>10} {sz:>10.1f}")

total_sz = sum(os.path.getsize(f'data/{f}') for f in files if os.path.exists(f'data/{f}')) / 1024 / 1024
print(f"{'TOTAL':<30} {'':>10} {total_sz:>10.1f}")

# Training pair breakdown
print("\n--- Training Pair Layer Breakdown ---")
pair_path = 'data/training_pairs.jsonl'
if os.path.exists(pair_path):
    layers = {}
    with open(pair_path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                layer = rec.get('type', rec.get('layer', 'unknown'))
                layers[layer] = layers.get(layer, 0) + 1
            except:
                pass
    for layer, cnt in sorted(layers.items(), key=lambda x: -x[1]):
        pct = cnt / sum(layers.values()) * 100
        print(f"  {layer:<35} {cnt:>8,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<35} {sum(layers.values()):>8,}")

# Vuln dataset field coverage
print("\n--- Vuln Dataset Field Coverage (sample 1000) ---")
ds_path = 'data/vuln_dataset.jsonl'
if os.path.exists(ds_path):
    field_counts = {}
    total = 0
    with open(ds_path) as fh:
        for i, line in enumerate(fh):
            if i >= 1000:
                break
            total += 1
            rec = json.loads(line)
            for k, v in rec.items():
                if v and v != [] and v != {} and v != '':
                    field_counts[k] = field_counts.get(k, 0) + 1
    for field, cnt in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"  {field:<35} {cnt:>5}/{total}  ({pct:.1f}%)")
