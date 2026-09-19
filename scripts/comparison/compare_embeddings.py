"""Compare existing method embeddings using the retained standardized protocols."""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['mousebrain','misar_seq','human_lymph_node','mouse_spleen','mouse_thymus','simulation','crc_stereocite','spatch'])
    parser.add_argument('--inputs',type=Path,required=True,help='JSON list of method and explicit source paths; see scripts/README.md.')
    parser.add_argument('--output-dir',type=Path,required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    from analysis.method_comparison import compare_embeddings
    result=compare_embeddings(args.dataset,json.loads(args.inputs.read_text()),args.output_dir)
    print(json.dumps(result,indent=2,default=str))
    return result


if __name__=='__main__':main()
