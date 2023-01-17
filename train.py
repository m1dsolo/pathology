from train_clam import train_CLAM
from train_dtfd import train_DTFD
from args import args

if __name__ == '__main__':
    if args['net']['type'] == 'DTFD':
        train_DTFD()
    else:
        train_CLAM()
