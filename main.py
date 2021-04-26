import argparse

from core.h36m_dataset import H36MDataset

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpath", type=str)
    parser.add_argument("--act", type=str, default="Directions-1")
    parser.add_argument("--subj", type=str, default="S1")
    parser.add_argument("--cam", type=int, default=55011271)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    dataset = H36MDataset(args.dpath)

    dataset.load_file(args.subj, args.act, args.cam)