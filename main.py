import argparse
import os
import time
import numpy as np

import deepod.utils as utils
import deepod.data as data
import deepod.models as models
import inspect


def dataset_list():
    members = inspect.getmembers(data)
    return [
        member[1]
        for member in members
        if inspect.isclass(member[1]) and member[1] != data.BaseDataset
    ]


def model_list():
    members = inspect.getmembers(models)
    return [
        member[1]
        for member in members
        if inspect.isclass(member[1]) and member[1] != data.BaseDataset
    ]


def supervised_list():
    return [models.DevNet, models.PReNet, models.DeepSAD, models.FeaWAD]


def unsupervised_list():
    return list(set(model_list()) - set(supervised_list()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="the path of the data sets")
    parser.add_argument("dataset", type=str, help="the class name of the data sets")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="how many times we repeat the experiments to obtain the average performance",
    )
    parser.add_argument(
        "--output_dir", type=str, default="@record/", help="the output file path"
    )
    parser.add_argument("--model", type=str, default="SLAD", help="Selected model")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cur_time = time.strftime("%m%d%H%M%S", time.localtime())
    result_file = os.path.join(
        args.output_dir,
        f"{cur_time}-{args.model}.{args.dataset}.csv",
    )

    if args.verbose:
        with open(result_file, "a") as f:
            print("\n---------------------------------------------------------", file=f)
            print(
                f"model: {args.model}, file: {args.data_dir}, "
                f"dataset: {args.dataset}, run: {args.runs}",
                file=f,
            )
            print("---------------------------------------------------------", file=f)
            print("data\t\tauc-roc\t\tstd\t\tauc-pr\t\tstd\t\tf1\t\tstd\t\ttime", file=f)

    avg_auc_lst, avg_ap_lst, avg_f1_lst = [], [], []

    dataset = None
    try:
        dataset = getattr(data, args.dataset)
    except AttributeError:
        print("This data set is not supported!")
        print("Supported Datasets: ")
        for i, clz in enumerate(dataset_list()):
            print(f"{i + 1}. f{clz.__class__.__name__}")
    model = None
    try:
        model = getattr(models, args.model)
    except AttributeError:
        print("This model is not supported!")
        print("Supported models: ")
        for i, clz in enumerate(model_list()):
            print(f"{i + 1}. f{clz.__class__.__name__}")

    train_set = dataset(args.data_dir, is_training=True, normalize="z-score")
    test_set = dataset(args.data_dir, is_training=False, normalize="z-score")

    auc_lst, ap_lst, f1_lst = (
        np.zeros(args.runs),
        np.zeros(args.runs),
        np.zeros(args.runs),
    )
    t1_lst, t2_lst = np.zeros(args.runs), np.zeros(args.runs)
    clf = None
    for i in range(args.runs):
        start_time = time.time()
        print(
            f"\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{args.dataset}]"
        )

        clf = model(epochs=50, random_state=42 + i)
        if model in supervised_list():
            clf.fit(train_set.data, y=train_set.binary_gts)
        else:
            clf.fit(train_set.data)
        train_time = time.time()
        scores = clf.decision_function(test_set.data)
        done_time = time.time()

        auc, ap, f1 = utils.evaluate(test_set.binary_gts, scores)
        auc_lst[i], ap_lst[i], f1_lst[i] = auc, ap, f1
        t1_lst[i] = train_time - start_time
        t2_lst[i] = done_time - start_time

        with open(result_file, "a") as f:
            txt = (
                f"{args.dataset:4}\t\t{auc_lst[i]:.4f}\t\tNull\t"
                f"{ap_lst[i]:.4f}\t\tNull\t{f1_lst[i]:.4f}\tNull\t"
                f"{t1_lst[i]:.1f}/{t2_lst[i]:.1f}"
            )
            print(txt)
            print(txt, file=f)

    avg_auc, avg_ap, avg_f1 = (
        np.average(auc_lst),
        np.average(ap_lst),
        np.average(f1_lst),
    )
    std_auc, std_ap, std_f1 = np.std(auc_lst), np.std(ap_lst), np.std(f1_lst)
    avg_time1 = np.average(t1_lst)
    avg_time2 = np.average(t2_lst)

    with open(result_file, "a") as f:
        txt = (
            f"{args.dataset:4}\t\t{avg_auc:.4f}\t\t{std_auc:.4f}\t"
            f"{avg_ap:.4f}\t\t{std_ap:.4f}\t{avg_f1:.4f}\t{std_f1:.4f}\t"
            f"{avg_time1:.1f}/{avg_time2:.1f}"
        )
        print(txt, file=f)
        print(txt)

    avg_auc_lst.append(avg_auc)
    avg_ap_lst.append(avg_ap)
    avg_f1_lst.append(avg_f1)


if __name__ == "__main__":
    main()
