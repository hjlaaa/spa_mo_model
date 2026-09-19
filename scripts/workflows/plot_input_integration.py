"""Create input-modality and integrated-embedding UMAP panels from existing results.

This post-hoc workflow may prepare source modality features. It does not run
SpaMosaic or train an integration model, and never rebuilds the training cache.
"""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['mousebrain','human_embryo','misar_seq','spatch','crc_stereocite','human_lymph_node','mouse_spleen','mouse_thymus','simulation'])
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--analysis-dir',required=True,type=Path,help='Existing scaler/cluster results for this run.')
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--max-samples',type=int,default=100000)
    parser.add_argument('--n-neighbors',type=int,default=30)
    parser.add_argument('--min-dist',type=float,default=.3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--overwrite-umap',action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    from analysis.input_integration import run_input_integration
    result=run_input_integration(args)
    print(json.dumps(result,indent=2,default=str))
    return result


if __name__=='__main__':main()
