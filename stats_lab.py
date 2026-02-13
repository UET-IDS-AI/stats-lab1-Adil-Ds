import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.title(f"Normal Distribution (n={n})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.title(f"Uniform Distribution (n={n})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.title(f"Bernoulli Distribution (n={n})")
    plt.xlabel("Outcome")
    plt.ylabel("Frequency")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    return sum(data) / len(data)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    n = len(data)
    if n < 2:
        return 0
    mu = sample_mean(data)
    # Sum of squared differences
    sq_diff = [(x - mu)**2 for x in data]
    return sum(sq_diff) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        
    q1 = sorted_data[int(0.25 * (n - 1))]
    q3 = sorted_data[int(0.75 * (n - 1))]
    
    return minimum, maximum, median, q1, q3



# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)



# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Sample variances
    var_x = np.sum((x - np.mean(x))**2) / (len(x) - 1)
    var_y = np.sum((y - np.mean(y))**2) / (len(y) - 1)

    # Sample covariance
    cov_xy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)

    # 2x2 covariance matrix
    return np.array([[var_x, cov_xy],
                     [cov_xy, var_y]])
