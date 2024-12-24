import multiprocessing
import numpy as np

def test(x, comb):
    if comb:
        return np.sqrt(np.sum(x))

def main():
    pool = multiprocessing.Pool() 
    pool = multiprocessing.Pool(processes=4) 
    inputs = [1,2,3,4] 
    input_zip = list(zip([np.array(inputs) for i in range(4)], [True for i in range(4)]))
    print(input_zip)
    outputs = pool.starmap(test, input_zip)
    pool.close()
    pool.join()
    print("Input: {}".format(inputs)) 
    print("Output: {}".format(outputs)) 

    '''N = 100
    pool = multiprocessing.Pool()
    result = pool.map(test, range(10,N))
    pool.close()
    pool.join()
    print("Program finished!")
    print(result)'''

if __name__ == "__main__":
    main()