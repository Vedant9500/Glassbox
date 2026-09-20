#!/usr/bin/env python3
"""
Post-processing utility to shrink large evolution pipeline logs.
Reduces file size by sampling generations, pruning populations, and removing large fields.
"""

import argparse
import gzip
import json
from pathlib import Path


def shrink_log(
    input_path: Path,
    output_path: Path,
    sample_every: int = 1,
    keep_top_k: int | None = None,
    elite_only: bool = False,
    skip_formulas: bool = False,
    compressed_output: bool = False,
) -> None:
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    # Determine if input is gzipped
    is_input_gz = input_path.suffix == ".gz"
    open_in = gzip.open if is_input_gz else open

    # Determine output path and compression
    if compressed_output and output_path.suffix != ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    open_out = gzip.open if output_path.suffix == ".gz" else open
    mode_out = "wt" if output_path.suffix == ".gz" else "w"

    count_in = 0
    count_out = 0
    malformed_lines = 0
    # §3.75: native C++ traces never carry is_elite — track whether ANY
    # individual in the file has the marker so --elite-only on a native
    # trace can warn loudly instead of silently emptying populations.
    seen_is_elite_marker = False

    print(f"Processing {input_path} -> {output_path}...")

    with (
        open_in(input_path, "rt", encoding="utf-8") as f_in,
        open_out(output_path, mode_out, encoding="utf-8") as f_out,
    ):
        for line in f_in:
            count_in += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Same counting rule as the §3.72 trace-analyzer fix: a
                # truncated tail record is expected, systematic corruption
                # must not masquerade as a smaller log.
                malformed_lines += 1
                continue

            event_type = event.get("event", "")

            # Special handling for generation.evaluate which contains the bulk of data
            if event_type in [
                "generation.evaluate",
                "generation.post_eval",
                "generation.post_reproduce",
                "init.refined",
            ]:
                gen = event.get("generation", 0)

                # Sampling logic
                is_sampled = (gen % sample_every == 0) or (event_type == "init.refined")

                if not is_sampled and event_type != "init.refined":
                    # In non-sampled generations, we might still want to keep the event
                    # but clear the population lists to save space
                    for key in ["population", "explorers"]:
                        cleared = event.get(key, [])
                        if any(
                            isinstance(ind, dict) and ind.get("is_elite", False)
                            for ind in cleared
                        ):
                            seen_is_elite_marker = True
                        event[key] = []
                    event["is_full_snapshot"] = False
                else:
                    # Sampled generation: apply pruning
                    for key in ["population", "explorers"]:
                        pop = event.get(key, [])
                        if not pop:
                            continue

                        # §3.74: preserve pre-prune sizes before shrinking.
                        event[key + "_size_original"] = len(pop)
                        processed_pop = []
                        for ind in pop:
                            # In C++, elites are the first N in some phases, but let's check for 'is_elite'
                            # Actually C++ doesn't have 'is_elite' in JSON by default, but we can assume
                            # the top K in any generation are the ones to keep.
                            if ind.get("is_elite", False):
                                seen_is_elite_marker = True
                            is_elite = ind.get("is_elite", False)

                            # Filtering logic
                            if elite_only and not is_elite:
                                continue

                            # Field stripping
                            if skip_formulas:
                                ind.pop("formula", None)
                                ind.pop("formula_error", None)

                            processed_pop.append(ind)

                        # Top-K pruning
                        if keep_top_k is not None:
                            # Sort by fitness (handling None and different field names)
                            def get_fitness(x):
                                # Try both 'fitness' (Python) and 'fitness' (C++) - oh wait they are both 'fitness'
                                return (
                                    x.get("fitness")
                                    if x.get("fitness") is not None
                                    else float("inf")
                                )

                            processed_pop.sort(key=get_fitness)
                            processed_pop = processed_pop[:keep_top_k]

                        event[key] = processed_pop
                        # §3.74: a pruned listing is NOT a full snapshot —
                        # downstream population-size/diversity analysis keys
                        # off this flag (evolution_pipeline_log:113-118).
                        event["is_full_snapshot"] = False

            # Write event
            f_out.write(json.dumps(event) + "\n")
            count_out += 1

    if malformed_lines:
        print(
            f"Warning: skipped {malformed_lines} malformed input line(s); "
            "a truncated tail record is expected, systematic corruption is not."
        )
    if elite_only and not seen_is_elite_marker:
        # §3.75: fail loud instead of silently emitting empty populations.
        print(
            "Warning: --elite-only matched zero individuals: no 'is_elite' "
            "marker exists in this trace (native C++ traces never emit one). "
            "Populations were emptied by the filter, not by evolution."
        )
    print(f"Done. Processed {count_in} lines into {count_out} lines.")
    size_in = input_path.stat().st_size / (1024 * 1024)
    size_out = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Size reduction: {size_in:.2f} MB -> {size_out:.2f} MB ({size_out / size_in * 100:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description="Shrink evolution pipeline logs")
    parser.add_argument("input", type=str, help="Input JSONL (.jsonl or .jsonl.gz)")
    parser.add_argument("output", type=str, help="Output JSONL")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep full population every N generations",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=None,
        help="Keep only top K individuals per population listing",
    )
    parser.add_argument(
        "--elite-only",
        action="store_true",
        help="Only keep individuals marked as elite (§3.75: native C++ traces "
        "never emit 'is_elite', so this empties their populations with a warning)",
    )
    parser.add_argument(
        "--skip-formulas",
        action="store_true",
        help="Remove 'formula' strings from all individuals",
    )
    parser.add_argument(
        "--compress", action="store_true", help="Compress output with gzip"
    )

    args = parser.parse_args()

    shrink_log(
        input_path=Path(args.input),
        output_path=Path(args.output),
        sample_every=args.sample_every,
        keep_top_k=args.keep_top_k,
        elite_only=args.elite_only,
        skip_formulas=args.skip_formulas,
        compressed_output=args.compress,
    )


if __name__ == "__main__":
    main()
