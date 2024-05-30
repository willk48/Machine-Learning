import pandas as pd
import matplotlib.pyplot as plt
'''Excel Funcs for quick accs
=SUM(A2,B3,C4,D5,E6,F7,G8,H9,I10,J11)
=SUM(A2:J11)
=DIVIDE(M2,N2)
=SUM(A2,B3)
=SUM(A2,B2,A3,B3)
=DIVIDE(E2,F2)
=SUM(A2,B3,C4)
=SUM(A2:C4)
=DIVIDE(F2,G2)'''

def gen_line_chart():
    MNIST_acc = [0.718875,0.7543333333,0.77775,0.7855]
    OCC_acc = [0.9878404669,0.9883268482,0.9908803502,0.9892996109]
    PENG_acc = [0.8394160584,0.9417475728,0.9854014599,0.9855072464]
    train_percs = [0.2,0.4,0.6,0.8]
    
    line_1=plt.plot(train_percs,MNIST_acc,label='mnist1000.csv')
    line_2=plt.plot(train_percs,OCC_acc,label='occupancy.csv')
    line_3=plt.plot(train_percs,PENG_acc,label='penguins.csv')
    
    leg = plt.legend(title="Dataset")
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Training Percentage')
    plt.title('Accuracy Over Different Train-Test Splits')
    plt.savefig('Q5_line_chart.png')

def main():
    gen_line_chart()

if __name__ == "__main__":
    main()