import torch
import train

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_agent = train.train()
