import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compute_A_null():
    #matrix = [male_mean,female_mean,non_smoker_mean,smoker_mean]
    A_null = [[0]*4 for p in range(48)] # A_null is plus

    for i in range(12):
        A_null[i][0]=1
        A_null[i][2]=1

    for i in range(12,24):
        A_null[i][0]=1
        A_null[i][3]=1

    for i in range(24,36):
        A_null[i][1]=1
        A_null[i][2]=1

    for i in range(36,48):
        A_null[i][1]=1
        A_null[i][3]=1
    # len(A_null) = 48

    return A_null

def compute_A():
    #matrix = [male_ns_mean,male_s_mean,female_ns_mean,female_s_mean]
    A=[[0]*4 for p in range(48)]
    for i in range(12):
        A[i][0]=1

    for i in range(12,24):
        A[i][1]=1

    for i in range(24,36):
        A[i][2]=1

    for i in range(36,48):
        A[i][3]=1

    # len(A) = 48
    return A


def compute_F_statistics(A_null, A, data, scaling_factor):
    I = np.identity(48)
    F_statistics=[]

    numerator = I - (np.matmul(np.matmul(A_null,np.linalg.pinv(np.matmul(A_null.T,A_null))),A_null.T))
    denominator = I - (np.matmul(np.matmul(A,np.linalg.pinv(np.matmul(A.T,A))),A.T))

    for index, row in data.iterrows():
        h = np.array(row.iloc[1:49].to_numpy().tolist())
        x1 = np.matmul(h.T, numerator)
        x2 = np.matmul(h.T, denominator)
        f = ((np.matmul(x1, h) / np.matmul(x2, h)) - 1) * scaling_factor
        F_statistics.append(f)

    return F_statistics


def main():

    data_path = "../data/"
    file_name = "Raw Data_GeneSpring.txt"
    data = pd.read_csv(os.path.join(data_path, file_name), delimiter = '\t')

    A_null = compute_A_null()
    A = compute_A()

    # calculate Ranks of A_null, A
    A = np.matrix(A) # returns the list of lists into a matrix form
    A_null = np.matrix(A_null)
    rank_A_null = np.linalg.matrix_rank(A_null) # Returning matrix A_null rank using SVD method
    rank_A = np.linalg.matrix_rank(A) # Returning matrix A rank using SVD method
    # rank_A_null
    # rank_A

    scaling_factor = (48-rank_A) / (rank_A-rank_A_null) # is degrees of freedom

    # Computing F_Statistics
    F_statistics = compute_F_statistics(A_null, A, data, scaling_factor)
    # print(F_statistics) # prints in matrix() form
    # print(len(F_statistics)) # 41093
    F_statistics = np.array(F_statistics)
    F_statistics = F_statistics.tolist()
    # print(F_statistics) # prints as list
    # print(type(F_statistics)) # list
    for i in range(len(F_statistics)):
        F_statistics[i] = F_statistics[i][0] # extracting values

    #calculating p_values
    p_values = 1 - stats.f.cdf(F_statistics, rank_A-rank_A_null, 48-rank_A)
    # print(p_values)
    print("p_values generated")
    plt.hist(p_values, bins=20)
    plt.show()
    print("Histogram plotted of p_values")

if __name__ == "__main__":
    main()
