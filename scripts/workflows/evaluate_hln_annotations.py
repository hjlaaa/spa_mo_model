"""Evaluate externally supplied HLN A1 manual labels against method embeddings."""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path,help='JSON list: method, paths, analysis_dir, annotation_path; see scripts/README.md.')
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--dry-run',action='store_true',help='Validate and calculate without writing analysis files.')
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    from analysis.hln_annotations import run_annotations
    result=run_annotations(json.loads(args.inputs.read_text()),args.output_dir,dry_run=args.dry_run)
    print(json.dumps(result,indent=2,default=str))
    return result


if __name__=='__main__':main()
