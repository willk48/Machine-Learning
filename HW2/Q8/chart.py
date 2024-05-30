import pandas as pd
import matplotlib.pyplot as plt

def gen_line_chart():
    tree_num = [10,20,30,40,50,60,70,80,90,100]
    acc_lst = [0.8972,0.926,0.9364,0.9376,0.936,0.9376,0.942,0.9432,0.9428,0.9448]
    
    line_1=plt.plot(tree_num,acc_lst)
    
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Tree Number')
    plt.title('Accuracy Over Different tree_num')
    plt.savefig('Q8_line_chart.png')

def main():
    gen_line_chart()

if __name__ == "__main__":
    main()