import argparse
import os
import time

import numpy as np
import deepod.data as data
import deepod.utils as utils
import deepod.models as models
import inspect


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="how many times we repeat the experiments to obtain the average performance",
    )
    parser.add_argument("data_dir", type=str, help="the path of the data sets")
    parser.add_argument("dataset", type=str, help="the class name of the data sets")
    parser.add_argument(
        "--output_dir", type=str, default="@record/", help="the output file path"
    )
    parser.add_argument("--model", type=str, default="SLAD", help="Selected model")
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(
        args.output_dir, f"{args.model}.{args.dataset}.csv"
    )
    cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())

    if args.verbose:
        f = open(result_file, "a")
        print("\n---------------------------------------------------------", file=f)
        print(
            f"model: {args.model}, file: {args.data_dir}, "
            f"datasets: {args.dataset}, {args.runs} runs, ",
            file=f,
        )
        print("---------------------------------------------------------", file=f)
        print("data, auc-roc, std, auc-pr, std, f1, std, time", file=f)
        f.close()

    avg_auc_lst, avg_ap_lst, avg_f1_lst = [], [], []

    dataset = None
    try:
        dataset = getattr(data, args.dataset)
    except AttributeError:
        members = inspect.getmembers(data)
        classes = [
            member[0]
            for member in members
            if inspect.isclass(member[1]) and member[1] != data.BaseDataset
        ]
        print("This data set is not supported!")
        print("Supported Datasets: ")
        for i, clz in enumerate(classes):
            print(f"{i + 1}. f{clz.__class__.__name__}")
    model = None
    try:
        model = getattr(models, args.model)
    except AttributeError:
        members = inspect.getmembers(models)
        classes = [member for member in members if inspect.isclass(member[1])]
        print("This model is not supported!")
        print("Supported models: ")
        for i, clz in enumerate(classes):
            print(f"{i + 1}. f{clz.__class__.__name__}")

    train_set = dataset(args.data_dir, is_training=True)
    test_set = dataset(args.data_dir, is_training=False)
    x_train, y_train = train_set.data, train_set.gts
    x_test, y_test = test_set.data, test_set.gts

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
        clf.fit(x_train, y=semi_y)
        train_time = time.time()
        scores = clf.decision_function(x_test)
        done_time = time.time()

        auc, ap, f1, _, _ = utils.cal_metrics(y_test, scores)
        auc_lst[i], ap_lst[i], f1_lst[i] = auc, ap, f1
        t1_lst[i] = train_time - start_time
        t2_lst[i] = done_time - start_time

        print(
            f"{args.dataset}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {f1_lst[i]:.4f}, "
            f"{t1_lst[i]:.1f}/{t2_lst[i]:.1f}, {args.model}"
        )

    avg_auc, avg_ap, avg_f1 = (
        np.average(auc_lst),
        np.average(ap_lst),
        np.average(f1_lst),
    )
    std_auc, std_ap, std_f1 = np.std(auc_lst), np.std(ap_lst), np.std(f1_lst)
    avg_time1 = np.average(t1_lst)
    avg_time2 = np.average(t2_lst)

    f = open(result_file, "a")
    txt = (
        f"{args.dataset}, "
        f"{avg_auc:.4f}, {std_auc:.4f}, "
        f"{avg_ap:.4f}, {std_ap:.4f}, "
        f"{avg_f1:.4f}, {std_f1:.4f}, "
        f"{avg_time1:.1f}/{avg_time2:.1f}"
    )
    print(txt, file=f)
    print(txt)
    f.close()

    avg_auc_lst.append(avg_auc)
    avg_ap_lst.append(avg_ap)
    avg_f1_lst.append(avg_f1)
