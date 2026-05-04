import json
with open('results/benchmark_20260504_113340.json') as f:
    d = json.load(f)
p11 = d['results'][10]
print("REASON:", p11.get('evolution_reason'))
print("MSE:", p11.get('mse'))
print("TIME:", p11.get('time'))
