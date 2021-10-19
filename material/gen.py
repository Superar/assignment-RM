import random
import sys

def main():
    argv = sys.argv[1:]
    n = int(argv[0])
    p = float(argv[1])
    s = int(argv[2])
    f = argv[3]

    random.seed(s)

    fout = open(f,"w")
    

    M = []
    m = 0
    for i in range(1,n):
        for j in range(i+1,n+1):
            if random.random() < p:
                m = m + 1
                M.append([i,j])
    fout.write(str(n) + " " + str(m) + "\n")
    for i in M:
        fout.write(str(i[0]) + " " + str(i[1]) + "\n")
    fout.close()



if __name__ == "__main__":
    main()



    



