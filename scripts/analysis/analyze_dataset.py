import os, json
from pathlib import Path
from collections import Counter, defaultdict

data_dir = Path('data')

# File sizes
print("=== DATA FILE SIZES ===")
for f in sorted(data_dir.glob('*')):
    if f.is_file():
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name:45s} {size_mb:8.2f} MB")

# Training pairs analysis
tp = data_dir / "training_pairs.jsonl"
if tp.exists():
    pairs = []
    with open(tp, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except:
                    pass
    
    print(f"\n=== TRAINING PAIRS: {len(pairs):,} total ===")
    
    # Layer distribution
    layers = Counter(p.get('layer','MISSING') for p in pairs)
    print("\nLayer distribution:")
    for layer, count in layers.most_common():
        pct = count/len(pairs)*100
        print(f"  {layer:45s} {count:>7,}  ({pct:5.1f}%)")
    
    # Type distribution (for cooccurrence)
    types = Counter(p.get('type','') for p in pairs if p.get('type'))
    if types:
        print("\nType distribution (where present):")
        for t, c in types.most_common():
            print(f"  {t:45s} {c:>7,}")
    
    # Output length stats
    output_lens = [len(p.get('output','')) for p in pairs]
    print(f"\nOutput length (chars): min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens)/len(output_lens):.0f}")
    
    # Check for empty fields
    empty_instruction = sum(1 for p in pairs if not p.get('instruction','').strip())
    empty_output = sum(1 for p in pairs if not p.get('output','').strip())
    empty_input = sum(1 for p in pairs if not p.get('input','').strip())
    print(f"\nEmpty fields: instruction={empty_instruction}, output={empty_output}, input={empty_input}")
    
    # Duplicate check
    seen = set()
    dupes = 0
    for p in pairs:
        key = (p.get('instruction','')[:120], p.get('output','')[:120])
        if key in seen:
            dupes += 1
        seen.add(key)
    print(f"Duplicates (by instruction+output prefix): {dupes}")
    
    # Sample a cooccurrence pair
    cooc = [p for p in pairs if p.get('layer') == 'vulnerability_cooccurrence']
    if cooc:
        print(f"\n=== SAMPLE CO-OCCURRENCE PAIR ===")
        s = cooc[0]
        print(f"  Type: {s.get('type','')}")
        print(f"  Input (first 200 chars): {s.get('input','')[:200]}")
        print(f"  Output (first 300 chars): {s.get('output','')[:300]}")
    
    corr = [p for p in pairs if p.get('layer') == 'vulnerability_correlation']
    if corr:
        print(f"\n=== SAMPLE CORRELATION PAIR ===")
        s = corr[0]
        print(f"  Instruction: {s.get('instruction','')[:200]}")
        print(f"  Output (first 300 chars): {s.get('output','')[:300]}")

# Co-occurrence raw data
cooc_file = data_dir / "raw_cooccurrence_v2.json"
if cooc_file.exists():
    with open(cooc_file) as f:
        cooc_data = json.load(f)
    stats = cooc_data.get('stats', {})
    print(f"\n=== RAW CO-OCCURRENCE V2 STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

# vuln_dataset.jsonl stats
vd = data_dir / "vuln_dataset.jsonl"
if vd.exists():
    count = 0
    with open(vd) as f:
        for line in f:
            if line.strip():
                count += 1
    print(f"\n=== vuln_dataset.jsonl: {count:,} records ===")
