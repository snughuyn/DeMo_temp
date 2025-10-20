from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from src.datamodule.ETRI_extractor import ETRIExtractor


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.pkl"))
    return scenario_files


def preprocess(args):
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        save_dir = Path("data/DeMo_ETRI_processed_lane") / mode
        extractor = ETRIExtractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            with multiprocessing.Pool(16) as p:
                list(tqdm(p.imap(extractor.save, scenario_files), total=len(scenario_files)))
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default="/home/yaaaaaaaang/data/ETRI_TP/raw")
    parser.add_argument("--parallel", "-p", action="store_true")

    args = parser.parse_args()
    preprocess(args)
