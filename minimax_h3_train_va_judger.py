import sys

from musubi_tuner.minimax_h3_train_va_judger import main


if __name__ == "__main__":
    main(["--stage", "train", *sys.argv[1:]])
