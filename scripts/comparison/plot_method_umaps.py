"""Plot method UMAPs aligned to an explicit reference cohort and saved joint labels."""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path,help='JSON list: method, dataset, reference_dir, analysis_dir.')
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--n-neighbors',type=int,default=30)
    parser.add_argument('--min-dist',type=float,default=.3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--overwrite-umap',action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    from analysis.method_umap import run_method_umaps
    result=run_method_umaps(json.loads(args.inputs.read_text()),args.output_dir,args)
    print(json.dumps(result,indent=2,default=str))
    return result


if __name__=='__main__':main()
