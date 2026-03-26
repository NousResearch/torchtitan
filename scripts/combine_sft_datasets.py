"""
Combine NousResearch/20260128_science_qa_speciale_filtered_unverified_packed (100%)
with NousResearch/science-math-code-combined at two ratios (50% and 100%),
after packing the latter with preprocess_data.py.

Usage:
    # Step 1: preprocess science-math-code-combined at 50% and 100%
    #   (run preprocess_data.py manually first, or pass --preprocess to do it here)

    python combine_sft_datasets.py \
        --ds2-50pct-path /path/to/science_math_code_packed_50pct \
        --ds2-100pct-path /path/to/science_math_code_packed_100pct \
        --output-dir /path/to/output \
        --tokenizer /path/to/tokenizer

    Or with --preprocess to run packing inline (requires --tokenizer):

    python combine_sft_datasets.py \
        --preprocess \
        --tokenizer /data/home/mormio/models/Qwen3-30B-A3B \
        --output-dir /home/shared/data/science_qa_plus_math_code
"""

import argparse
import os
import random
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.dataset as pa_ds

SCHEMA = pa.schema(
    [
        pa.field("inputs", pa.large_list(pa.int32())),
        pa.field("labels", pa.large_list(pa.int32())),
        pa.field("position_ids", pa.large_list(pa.int32())),
        pa.field("sequence_lengths", pa.large_list(pa.int64())),
    ]
)

DATASET_INFO = r"""{
  "citation": "",
  "description": "",
  "features": {
    "inputs": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "labels": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "position_ids": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "sequence_lengths": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "LargeList"
    }
  },
  "homepage": "",
  "license": ""
}"""


PREPROCESS_PYTHON = "/home/tinuviel/torchtitan/.venv/bin/python"


def run_preprocess(dataset, output_path, tokenizer, epochs, limit=None, seed=42):
    """Run preprocess_data.py to pack a HuggingFace dataset."""
    script = os.path.join(os.path.dirname(__file__), "preprocess_data.py")
    python = PREPROCESS_PYTHON if os.path.exists(PREPROCESS_PYTHON) else sys.executable
    cmd = [
        python,
        script,
        "--dataset",
        dataset,
        "--tokenizer",
        tokenizer,
        "--chat",
        "--pack-to-sequence-length",
        "65536",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--save-to-disk",
        output_path,
        "--num-proc",
        "16",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print(f"\n{'='*60}")
    print(f"Running preprocess_data.py:")
    print(f"  dataset: {dataset}" + (f" (limit={limit})" if limit else ""))
    print(f"  output:  {output_path}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def save_combined_as_arrow(combined_ds, output_dir, seed=42):
    """
    Save a HuggingFace Dataset as the custom arrow format used by preprocess_data.py.
    Shuffles before saving so the two datasets are interleaved.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Shuffling {len(combined_ds)} combined samples (seed={seed})...")
    combined_ds = combined_ds.shuffle(seed=seed)

    # Write in chunks (1 arrow file per shard of 1000 samples for manageability)
    shard_size = 1000
    num_shards = (len(combined_ds) + shard_size - 1) // shard_size
    file_counter = 0

    print(f"Writing {num_shards} arrow shards to {output_dir}...")
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(combined_ds))
        shard = combined_ds.select(range(start, end))

        oriented_data = {
            "inputs": [list(x) for x in shard["inputs"]],
            "labels": [list(x) for x in shard["labels"]],
            "position_ids": [list(x) for x in shard["position_ids"]],
            "sequence_lengths": [list(x) for x in shard["sequence_lengths"]],
        }
        pa_table = pa.Table.from_pydict(oriented_data, schema=SCHEMA)

        tmp_dir = os.path.join(output_dir, f"_tmp_{shard_idx}")
        pa_ds.write_dataset(pa_table, tmp_dir, format="arrow")
        src = os.path.join(tmp_dir, "part-0.arrow")
        dst = os.path.join(output_dir, f"data-{file_counter:05d}-of-TOTAL.arrow")
        shutil.move(src, dst)
        shutil.rmtree(tmp_dir)
        file_counter += 1

        if (shard_idx + 1) % 10 == 0:
            print(f"  {shard_idx + 1}/{num_shards} shards written...")

    # Rename TOTAL placeholders
    for i in range(file_counter):
        old = os.path.join(output_dir, f"data-{i:05d}-of-TOTAL.arrow")
        new = os.path.join(output_dir, f"data-{i:05d}-of-{file_counter:05d}.arrow")
        os.rename(old, new)

    with open(os.path.join(output_dir, "dataset_info.json"), "wb") as f:
        f.write(DATASET_INFO.encode())

    print(f"Saved {file_counter} arrow files to {output_dir}")


def combine_and_save(ds1_path_or_hf, ds2_path, output_dir, label, seed=42):
    from datasets import concatenate_datasets, load_dataset

    print(f"\n{'='*60}")
    print(f"Combining datasets → {label}")
    print(f"  ds1: {ds1_path_or_hf}")
    print(f"  ds2: {ds2_path}")
    print(f"  out: {output_dir}")
    print(f"{'='*60}")

    print("Loading dataset 1 (science_qa packed)...")
    ds1 = load_dataset(ds1_path_or_hf, split="train")
    print(f"  ds1 rows: {len(ds1)}")

    print("Loading dataset 2 (science-math-code packed)...")
    ds2 = load_dataset(ds2_path, split="train")
    print(f"  ds2 rows: {len(ds2)}")

    # Align columns (ds1 from HF may have different dtypes)
    ds1 = ds1.select_columns(["inputs", "labels", "position_ids", "sequence_lengths"])
    ds2 = ds2.select_columns(["inputs", "labels", "position_ids", "sequence_lengths"])

    combined = concatenate_datasets([ds1, ds2])
    print(f"  combined rows: {len(combined)}")
    steps = len(combined) // 64
    print(f"  => ~{steps} steps at global_batch_size=64")

    save_combined_as_arrow(combined, output_dir, seed=seed)

    # Verify
    final = load_dataset(output_dir)
    print(f"Verified: {len(final['train'])} rows readable from {output_dir}")
    return len(combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocess_data.py for science-math-code-combined (50% and 100%)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/data/home/mormio/models/Qwen3-30B-A3B",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of epochs to pack dataset 2 for"
    )
    parser.add_argument(
        "--ds1-hf",
        type=str,
        default="NousResearch/20260128_science_qa_speciale_filtered_unverified_packed",
        help="HuggingFace dataset ID for the packed science_qa dataset (dataset 1)",
    )
    parser.add_argument(
        "--ds2-50pct-path",
        type=str,
        default=None,
        help="Path to pre-packed science-math-code at 50%% (if --preprocess not used)",
    )
    parser.add_argument(
        "--ds2-100pct-path",
        type=str,
        default=None,
        help="Path to pre-packed science-math-code at 100%% (if --preprocess not used)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write combined datasets into (two subdirs will be created)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds2_50_path = args.ds2_50pct_path or os.path.join(
        args.output_dir, "science_math_code_packed_50pct"
    )
    ds2_100_path = args.ds2_100pct_path or os.path.join(
        args.output_dir, "science_math_code_packed_100pct"
    )

    # Step 1: optionally preprocess dataset 2
    if args.preprocess:
        ds2_hf = "NousResearch/science-math-code-combined"
        n_total = 61311  # known total row count

        if not os.path.exists(ds2_50_path):
            run_preprocess(
                dataset=ds2_hf,
                output_path=ds2_50_path,
                tokenizer=args.tokenizer,
                epochs=args.epochs,
                limit=n_total // 2,  # 50%: ~30655 samples
                seed=args.seed,
            )
        else:
            print(f"Skipping 50% preprocess — {ds2_50_path} already exists")

        if not os.path.exists(ds2_100_path):
            run_preprocess(
                dataset=ds2_hf,
                output_path=ds2_100_path,
                tokenizer=args.tokenizer,
                epochs=args.epochs,
                limit=None,  # 100%
                seed=args.seed,
            )
        else:
            print(f"Skipping 100% preprocess — {ds2_100_path} already exists")

    # Step 2: combine at each ratio
    out_50 = os.path.join(args.output_dir, "combined_scienceqa_100pct_mathcode_50pct")
    out_100 = os.path.join(args.output_dir, "combined_scienceqa_100pct_mathcode_100pct")

    rows_50 = combine_and_save(
        args.ds1_hf,
        ds2_50_path,
        out_50,
        label="100% sci_qa + 50% math_code",
        seed=args.seed,
    )
    rows_100 = combine_and_save(
        args.ds1_hf,
        ds2_100_path,
        out_100,
        label="100% sci_qa + 100% math_code",
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"  Run 1 (50%):  {out_50}")
    print(f"    => {rows_50 // 64} steps at global_batch_size=64")
    print(f"  Run 2 (100%): {out_100}")
    print(f"    => {rows_100 // 64} steps at global_batch_size=64")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
