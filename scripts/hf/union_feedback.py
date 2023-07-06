import pickle
import os
import argparse


def main(args):
    with open(args.feedback_paths[0], "rb") as f:
        file = pickle.load(f)
    print(args.feedback_paths)
    print("\n")
    for i in range(len(args.feedback_paths) - 1):
        with open(args.feedback_paths[i + 1], "rb") as f:
            up_file = pickle.load(f)
        file.update(up_file)

    value_list = [i for i in file.values()]
    count = [value_list.count(i) for i in range(2)]
    count.append(len(value_list))
    print("label 0 : ", count[0])
    print("\nlabel 1 : ", count[1])
    print("\ntotal feedback count :", count[2])

    with open(os.path.join(args.out_union_feedback_dir, f"union_{count[0]}_{count[1]}.pkl"), "wb") as f:
        pickle.dump(file, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_paths", type=str, nargs='+',
                        help = "list of path to each feedback pickle files"
    )
    parser.add_argument("--out_union_feedback_dir", type=str,
                        help = "directory where you want to save union feedback"
    )
    args = parser.parse_args()
    main(args)