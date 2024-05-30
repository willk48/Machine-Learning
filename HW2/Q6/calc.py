import sys
from math import sqrt
from statsmodels.stats.proportion import proportion_confint
#found from the internet at https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
#lower, upper = proportion_confint(88, 100, 0.05)
#print('lower=%.3f, upper=%.3f' % (lower, upper))

#usage python calc.py correct_pred total 

correct = float(sys.argv[1])
total_n = float(sys.argv[2])

def acc(corr, total):
    return corr/total

def se(acc,n):
    return sqrt((acc*(1-acc))/n)

def ci(corr,total):
    lower, upper = proportion_confint(corr, total, 0.05)
    print('lower=%.3f, upper=%.3f' % (lower, upper))

def main():
    acc_check = acc(correct,total_n)
    print(f'ACC: {acc_check}')
    print(f'SE: {se(acc_check,total_n)}')
    ci(correct,total_n)

if __name__ == "__main__":
    main()