import os
import sys
import argparse

try:
    from dataloder import get_dataloader
    from eval import evaluator
    from bert_runner import get_runner
except ModuleNotFoundError:
    from .dataloder import get_dataloader
    from .eval import evaluator
    from .bert_runner import get_runner

sys.path.insert(0, os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../../dataset/squad/dev-v1.1.json", help="data file path")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--count", type=int, default=50, help="Maximum number of examples to consider")
    parser.add_argument("--model", type=str, help="bmodel path")
    parser.add_argument("--cache_path", default='eval_features.pickle', help="pickle path")
    parser.add_argument("--out_file", default='./result/predictions.json', help='output json path')
    parser.add_argument("--devices", '-d',type=int, nargs='*', help='Devices',default=[0])
    args = parser.parse_args()
    return args

def run(model, data, count, cache_path, out_file, devices, accuracy = True):
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.model = model
    args.data = data
    args.count = count
    args.accuracy = accuracy

    args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab.txt')
    args.cache_path = cache_path
    args.out_file = out_file
    args.devices = devices
    config = {
        "accuracy": accuracy,
        "total_count": count,
    }

    runner = get_runner(args)
    squad_dl = get_dataloader(count_override=args.count,
                              cache_path=args.cache_path,
                              input_file=args.data,
                              load_fn=runner.run_one_query)
    squad_dl.start_test(config)
    result = runner.runner_result
    if args.accuracy:
        return evaluator(args, result)

def main():
    if not os.path.exists("result"):
        os.makedirs("result")
    args = get_args()

    runner = get_runner(args)
    # warmup
    # runner.run_one_item()

    config = {
        "accuracy": args.accuracy,
        "total_count": args.count,
    }
    squad_dl = get_dataloader(count_override=args.count,
                              cache_path=args.cache_path,
                              input_file=args.data,
                              load_fn=runner.run_one_query)
    squad_dl.start_test(config)
    result = runner.runner_result
    if args.accuracy:
        result = evaluator(args, result)
    print(result)
    print("Done!")

if __name__ == "__main__":
    main()
