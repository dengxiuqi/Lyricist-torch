import torch
from data import Lyric
from config import fix_length, output_size
from utils import train


learning_rate = 5e-4
total_epoch = 51
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    singers = ["周杰伦", "李荣浩", "林俊杰", "许嵩", "陈粒"]
    for singer in singers:
        print("=" * 10)
        print("fine tune:" + singer)
        dataset = Lyric(batch_size=batch_size,
                        fix_length=fix_length,
                        singer=singer,
                        target_vocab_size=output_size,
                        device=device)
        train(dataset, learning_rate, total_epoch, device, save_epoch=total_epoch-1, log_step=5, test_epoch=5)
