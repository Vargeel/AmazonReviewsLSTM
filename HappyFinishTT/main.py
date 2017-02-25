import getopt,sys
from training_net import train
from testing_net import test



def main(argv):
    trainPath = ''
    testPath = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["test=","train="])
    except getopt.GetoptError:
        print('usage : main.py --train <trainPath>  OR --test <testPath>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("--help","-h"):
            print('usage : main.py --train <trainPath>  OR --test <testPath>')
            sys.exit()
        elif opt in ("--train"):
            trainPath = arg
        elif opt in ("--test"):
            testPath = arg
    if trainPath != '' and testPath!= '':
        print('usage : main.py --train <trainPath>  OR --test <testPath>')
        sys.exit()
    if trainPath:
        train(trainPath)
    elif testPath:
        test(testPath)
    else:
        print('usage : main.py --train <trainPath>  OR --test <testPath>')


if __name__ == "__main__":
   main(sys.argv[1:])