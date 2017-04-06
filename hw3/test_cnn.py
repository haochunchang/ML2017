import numpy as np
import sys
import feature
import keras

def nn_test(test, outfilepath):
    
    # Loading in testing data
    
    # Preprocessing

    # Loading in trained model
    
    with open(outfilepath, 'w') as o:
        o.write("id,label\n")
        
        for i in range(len(test)):   
            o.write(str(i+1)+","+str(y[i]))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)

if __name__ == "__main__":
    nn_test(sys.argv[1], sys.argv[2]) 
