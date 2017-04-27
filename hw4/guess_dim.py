import sys

def train_guess(data):
    pass


def guess_dim(data, w, outfilepath):
    with open(outfilepath, 'w') as o:
        o.write("SetId,LogDim\n") 
        
        # Execute out predict function

        for i in range(len(test)):   
            o.write(str(i)+","+str(y[i]))
            o.write("\n")
    print("Testing result stored in %s" % outfilepath)


def main(infilepath, outfilepath):
    w = train_guess(data)
    guess_dim(data, w, outfilepath)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
