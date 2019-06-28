import torch
from data import Lyric
from config import fix_length, output_size
from utils import train


learning_rate = 5e-4
total_epoch = 51
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = Lyric(batch_size=batch_size,
                fix_length=fix_length,
                target_vocab_size=output_size,
                device=device)


if __name__ == "__main__":
    train(dataset, learning_rate, total_epoch, device)
