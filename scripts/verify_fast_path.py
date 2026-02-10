
import sys
import os
import time
import subprocess
import re
import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

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
    {
        "name": "Constant Hunter",
        "formula": "pi * x**2 + sqrt(2) * sin(x) - 1.618",
        "args": "--x-min -3 --x-max 3 --n-samples 400 --generations 30 --population 25 --ops-periodic --ops-power"
    },
    # Multi-input benchmarks
    {
        "name": "2D Sum of Squares",
        "formula": "x0^2 + x1^2",
        "args": "--x-min -3 --x-max 3 --n-samples 400 --generations 10 --population 20",
        "n_inputs": 2
    },
    {
        "name": "2D Cross Term",
        "formula": "x0*x1 + x0 + x1",
        "args": "--x-min -3 --x-max 3 --n-samples 400 --generations 10 --population 20",
        "n_inputs": 2
    },
    {
        "name": "2D Mixed Polynomial",
        "formula": "x0^2 + 2*x0*x1 + x1^2",
        "args": "--x-min -2 --x-max 2 --n-samples 400 --generations 15 --population 25",
        "n_inputs": 2
    },
    # Phase 1 constant detection benchmarks
    {
        "name": "Non-Integer Power",
        "formula": "x**2.5 + sin(x)",
        "args": "--x-min 0.1 --x-max 5 --n-samples 400 --ops-periodic --ops-power"
    },
    {
        "name": "Symbolic Coefficients",
        "formula": "sqrt(2)*sin(x) + pi*x",
        "args": "--x-min -3 --x-max 3 --n-samples 400 --ops-periodic --ops-power"
    },
    {
        "name": "Multi-Frequency FFT",
        "formula": "sin(3*x) + sin(7*x)",
        "args": "--x-min -5 --x-max 5 --n-samples 500 --ops-periodic --no-ops-exp --no-ops-log"
    },
]

def run_test(benchmark, model_path=None):
    print(f"\n============================================================")
    print(f"TESTING: {benchmark['name']}")
    print(f"TARGET:  {benchmark['formula']}")
    print(f"============================================================")
    
    model_arg = f"--curve-classifier-model \"{model_path}\"" if model_path else ""
    n_inputs_arg = f"--n-inputs {benchmark.get('n_inputs', 1)}"
    cmd = f"python scripts/sr_tester.py --mode single --formula \"{benchmark['formula']}\" --curve-classifier {model_arg} {n_inputs_arg} --no-viz {benchmark['args']}"
    
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

def list_models():
    """List available models in the models directory."""
    models = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.iterdir():
            if f.suffix in ('.pt', '.pkl', '.joblib'):
                models.append(f)
    return models


def main():
    parser = argparse.ArgumentParser(description="Fast-Path Verification Suite")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to curve classifier model (default: auto-detect)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Run only a specific benchmark by name (partial match)")
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for m in list_models():
            size_mb = m.stat().st_size / (1024 * 1024)
            print(f"  {m.name:<30} ({size_mb:.1f} MB)")
        return
    
    # Resolve model path
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            # Try looking in models directory
            model_path = MODELS_DIR / args.model
        if not model_path.exists():
            print(f"Error: Model not found: {args.model}")
            print("Available models:")
            for m in list_models():
                print(f"  {m.name}")
            return
        print(f"Using model: {model_path}")
    
    # Filter benchmarks if requested
    benchmarks = BENCHMARKS
    if args.benchmark:
        benchmarks = [b for b in BENCHMARKS if args.benchmark.lower() in b['name'].lower()]
        if not benchmarks:
            print(f"No benchmarks matching '{args.benchmark}'")
            print("Available benchmarks:")
            for b in BENCHMARKS:
                print(f"  {b['name']}")
            return
    
    print("Starting Fast-Path Verification Suite...")
    print(f"Testing {len(benchmarks)} benchmarks")
    if model_path:
        print(f"Model: {model_path.name}")
    
    results = []
    for bench in benchmarks:
        results.append(run_test(bench, model_path))
        
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
