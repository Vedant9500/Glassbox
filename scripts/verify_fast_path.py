
import sys
import os
import time
import subprocess
import re

BENCHMARKS = [
    {
        "name": "Nguyen-7 (Logarithmic)",
        "formula": "log(x+1) + log(x**2+1)",
        "args": "--x-min 0.1 --x-max 4 --full-ops --no-ops-periodic"
    },
    {
        "name": "Damped Sine (Oscillator)",
        "formula": "exp(-0.2*x) * sin(5*x)",
        "args": "--x-min 0 --x-max 10 --full-ops --ops-periodic --ops-exp"
    },
    {
        "name": "Rational (Lorentzian)",
        "formula": "x / (x**2 + 1)",
        "args": "--x-min -5 --x-max 5 --no-ops-periodic --no-ops-exp --no-ops-log"
    },
    {
        "name": "Nguyen-10 (Composition - Hard)",
        "formula": "sin(x) + sin(x**2)",
        "args": "--x-min 0.1 --x-max 3 --n-samples 500 --ops-periodic --ops-power"
    },
    {
        "name": "Relativistic Mass",
        "formula": "1 / sqrt(1 - x**2)",
        "args": "--x-min -0.85 --x-max 0.85 --n-samples 300 --generations 25 --population 20 --no-ops-periodic --no-ops-exp --sample-avoid 1 -1 --sample-epsilon 0.15"
    },
    {
        "name": "Planck's Law (Simplified)",
        "formula": "x**3 / (exp(x) - 1)",
        "args": "--x-min 0.2 --x-max 5 --n-samples 400 --full-ops --generations 50 --population 30 --no-ops-periodic --ops-exp --sample-avoid 0 --sample-epsilon 0.2"
    },
    # {
    #     "name": "Damped Oscillator",
    #     "formula": "exp(-0.1*x) * cos(2*x)",
    #     "args": "--x-min 0 --x-max 10 --n-samples 400 --full-ops --ops-periodic --ops-exp"
    # },
    # {
    #     "name": "Nesting Doll",
    #     "formula": "sin(cos(x)) + exp(sin(x))",
    #     "args": "--x-min -3.14 --x-max 3.14 --n-samples 500 --full-ops --generations 60 --population 40 --ops-periodic --ops-exp"
    # },
    {
        "name": "Constant Hunter",
        "formula": "pi * x**2 + sqrt(2) * sin(x) - 1.618",
        "args": "--x-min -3 --x-max 3 --n-samples 400 --generations 30 --population 25 --ops-periodic --ops-power"
    }
]

def run_test(benchmark):
    print(f"\n============================================================")
    print(f"TESTING: {benchmark['name']}")
    print(f"TARGET:  {benchmark['formula']}")
    print(f"============================================================")
    
    cmd = f"python scripts/sr_tester.py --mode single --formula \"{benchmark['formula']}\" --curve-classifier --no-viz {benchmark['args']}"
    
    start_time = time.time()
    try:
        # Use utf-8 and ignore errors to handle Windows console special chars
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='ignore'
        )
        output = result.stdout + "\n" + result.stderr
        
        # Extract results
        mse_match = re.search(r"FINAL MSE:\s+([0-9\.e\-]+)", output)
        formula_match = re.search(r"DISCOVERED:\s+(.+)", output)
        time_match = re.search(r"TOTAL TIME:\s+([0-9\.]+)", output)
        
        mse = float(mse_match.group(1)) if mse_match else float('inf')
        discov_formula = formula_match.group(1) if formula_match else "Failed"
        elapsed = float(time_match.group(1)) if time_match else (time.time() - start_time)
        
        # Debug: Print output if MSE is inf or nan
        if mse == float('inf') or mse != mse:
            print("\n!!! DEBUG: RAW OUTPUT START !!!")
            print(output)
            print("!!! DEBUG: RAW OUTPUT END !!!\n")
            
        print(f"RESULT: {'PASS' if mse < 0.001 else 'FAIL'}")
        print(f"MSE:    {mse:.6f}")
        print(f"Time:   {elapsed:.2f}s")
        print(f"Found:  {discov_formula}")
        
        return {
            "name": benchmark["name"],
            "pass": mse < 0.001,
            "mse": mse,
            "time": elapsed,
            "formula": discov_formula
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"name": benchmark["name"], "pass": False, "mse": -1, "time": 0, "formula": "Error"}

def main():
    print("Starting Fast-Path Verification Suite...")
    print(f"Testing {len(BENCHMARKS)} benchmarks")
    
    results = []
    for bench in BENCHMARKS:
        results.append(run_test(bench))
        
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'BENCHMARK':<30} | {'STATUS':<6} | {'MSE':<10} | {'TIME':<8} | {'FORMULA'}")
    print("-" * 80)
    
    passed = 0
    for r in results:
        status = "PASS" if r['pass'] else "FAIL"
        if r['pass']: passed += 1
        # Truncate formula if too long
        f_disp = (r['formula'][:40] + '...') if len(r['formula']) > 40 else r['formula']
        print(f"{r['name']:<30} | {status:<6} | {r['mse']:<10.6f} | {r['time']:<8.2f} | {f_disp}")
    
    print("-" * 80)
    print(f"Total Passed: {passed}/{len(BENCHMARKS)}")

if __name__ == "__main__":
    main()
