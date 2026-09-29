import argparse

from . import config
from .data import download_data
from .deploy import promote_model
from .evaluate import evaluate_model, should_deploy
from .preprocess import preprocess
from .train import train_model


def run_all(n_estimators: int, seed: int, force_deploy: bool) -> None:
    download_data()
    preprocess()
    train_model(n_estimators=n_estimators, seed=seed)
    metrics = evaluate_model()
    if force_deploy or should_deploy(metrics["accuracy"]):
        promote_model()
    else:
        print("Accuracy did not improve - keeping current production model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML pipeline CLI")
    parser.add_argument("command", choices=["download", "preprocess", "train", "evaluate", "all"])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-deploy", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        download_data()
    elif args.command == "preprocess":
        preprocess()
    elif args.command == "train":
        train_model(n_estimators=args.n_estimators, seed=args.seed)
    elif args.command == "evaluate":
        evaluate_model()
    else:
        run_all(args.n_estimators, args.seed, args.force_deploy)
    print(f"(project root: {config.ROOT})")


if __name__ == "__main__":
    main()
