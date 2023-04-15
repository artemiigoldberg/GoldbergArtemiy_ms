import math
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats
from scipy.special import factorial
import matplotlib.pyplot as plt
import seaborn as sns
import os

numBins = 20
distType = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]

def getDistribution(distType, num):
    if distType == "Normal":
        return np.random.normal(0, 1, num)
    elif distType == "Cauchy":
        return np.random.standard_cauchy(num)
    elif distType == "Laplace":
        return np.random.laplace(0, 1 / np.sqrt(2), num)
    elif distType == "Poisson":
        return np.random.poisson(10, num)
    elif distType == "Uniform":
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), num)
    return []

def getDensityFunc(distType, array):
    if distType == "Normal":
        return [1 / (np.sqrt(2 * np.pi)) * np.exp(-1 * x * x / 2) for x in array]
    elif distType == "Cauchy":
        return [1 / (np.pi * (x * x + 1))  for x in array]
    elif distType == "Laplace":
        return [1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(x))  for x in array]
    elif distType == "Poisson":
        return [np.power(10, x) * np.exp(-10) / factorial(x) for x in array]
    elif distType == "Uniform":
        return [1 / (2 * np.sqrt(3)) if np.fabs(x) <= np.sqrt(3) else 0  for x in array]
    return []

def task1():
    N = [10, 50, 100]
    for distribution in distType:
        figure, axes = plt.subplots(1, 3, figsize = (11.8, 3.9))
        plt.subplots_adjust(wspace=0.5)
        figure.suptitle(distribution + " distribution", y = 1, fontsize = 20)
        for i in range(len(N)):
            x = getDistribution(distribution, N[i])
            n, bins, patches = axes[i].hist(x, numBins, density = 1, edgecolor = "blue", alpha = 0.3)
            axes[i].plot(bins, getDensityFunc(distribution, bins), color = "red");
            axes[i].set_title("n = " + str(N[i]))
        if not os.path.isdir("MathStat"):
          os.makedirs("MathStat")
        plt.savefig("MathStat/" + distribution + ".png")
    plt.show()

def calculateMean(x):
    return np.mean(x)

def calculateVar(x):
    return np.var(x)

def calculateMedian(x):
    return np.median(x)

def calculateZR(x):
    return (min(x) + max(x)) / 2

def calculateQuantile(x, index):
    return np.quantile(x, index)

def calculateZQ(x):
    return (calculateQuantile(x, 0.25) + calculateQuantile(x, 0.75)) / 2

def calculateZTR(x):
    r = int(len(x) * 0.25)
    sX = np.sort(x)
    sum = 0
    for i in range(r + 1, len(x) - r):
        sum += sX[i]
    return sum / (len(x) - 2 * r)

def task2():
    N = [10, 50, 100]
    numOfRepeats = 1000

    for distribution in distType:
        result = []
        for size in N:
            mean = []
            median = []
            ZR = []
            ZQ = []
            ZTR = []
            for i in range(0, numOfRepeats):
                x = getDistribution(distribution, size)
                mean.append(calculateMean(x))
                median.append(calculateMedian(x))
                ZR.append(calculateZR(x))
                ZQ.append(calculateZQ(x))
                ZTR.append(calculateZTR(x))
            result.append(distribution + ":")
            result.append("N = " + str(size))
            result.append([" E(z) " + str(size),
                           "x_: " + str(np.around(calculateMean(mean), decimals = 6)),
                           "med: " + str(np.around(calculateMean(median), decimals = 6)),
                           "ZR: " + str(np.around(calculateMean(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateMean(ZQ), decimals = 6)),
                           "ZTR: " + str(np.around(calculateMean(ZTR), decimals = 6))])
            result.append([" D(z) " + str(size),
                           "x_: " + str(np.around(calculateVar(mean), decimals = 6)),
                           "med: " + str(np.around(calculateVar(median), decimals = 6)),
                           "ZR :" + str(np.around(calculateVar(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateVar(ZQ), decimals = 6)),
                           "ZTR :" + str(np.around(calculateVar(ZTR), decimals = 6))])
            result.append([" E(z) - sqrt(D(z)) " + str(size),
                           "x_: " + str(np.around(calculateMean(mean) - np.std(mean), decimals = 6)),
                           "med: " + str(np.around(calculateMean(median) - np.std(median), decimals = 6)),
                           "ZR :" + str(np.around(calculateMean(ZR) - np.std(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateMean(ZQ) - np.std(ZQ), decimals = 6)),
                           "ZTR :" + str(np.around(calculateMean(ZTR) - np.std(ZTR), decimals = 6))])
            result.append([" E(z) + sqrt(D(z)) " + str(size),
                           "x_: " + str(np.around(calculateMean(mean) + np.std(mean), decimals = 6)),
                           "med: " + str(np.around(calculateMean(median) + np.std(median), decimals = 6)),
                           "ZR :" + str(np.around(calculateMean(ZR) + np.std(ZR), decimals = 6)),
                           "ZQ: " + str(np.around(calculateMean(ZQ) + np.std(ZQ), decimals = 6)),
                           "ZTR :" + str(np.around(calculateMean(ZTR) + np.std(ZTR), decimals = 6))])
        if not os.path.isdir("task2"):
            os.makedirs("task2")
        fileName = distribution + "_data"
        completeName = os.path.join("task2/", fileName + ".txt")
        file = open(completeName, "w")
        for element in result:
            file.write(str(element) + "\n")
        file.close()

def task3():
    N = [20, 100]
    numOfRepeats = 1000
    #boxplot
    for distribution in distType:
        x20 = getDistribution(distribution, 20)
        x100 = getDistribution(distribution, 100)
        plt.boxplot((x20, x100), labels = ["n = 20", "n = 100"])
        plt.ylabel("X")
        plt.title(distribution)
        if not os.path.isdir("task3"):
          os.makedirs("task3")
        plt.savefig("task3/" + distribution + ".png")
        plt.figure()
    #outliers
    result = []
    for distribution in distType:
        for size in N:
            count = 0
            for i in range(numOfRepeats):
                x = getDistribution(distribution, size)

                min = calculateQuantile(x, 0.25) - 1.5 * (calculateQuantile(x, 0.75) - calculateQuantile(x, 0.25))
                max = calculateQuantile(x, 0.75) + 1.5 * (calculateQuantile(x, 0.75) - calculateQuantile(x, 0.25))

                for j in range(size):
                    if x[j] > max or x[j] < min:
                        count += 1
            count /= numOfRepeats
            result.append(distribution + " n = " + str(size) + " number of outliers = " + str(np.around(count / size, decimals = 3)))
    if not os.path.isdir("task3"):
      os.makedirs("task3")
    completeName = os.path.join("task3/", "outliers.txt")
    file = open(completeName, "w")
    for element in result:
      file.write(str(element) + "\n")
    file.close()

def getCDF(distType, array):
    if distType == "Normal":
        return stats.norm.cdf(array)
    elif distType == "Cauchy":
        return stats.cauchy.cdf(array)
    elif distType == "Laplace":
        return stats.laplace.cdf(array)
    elif distType == "Poisson":
        return stats.poisson.cdf(array, 10)
    elif distType == "Uniform":
        return stats.uniform.cdf(array)
    return []

def getPDF(distType, array):
    if distType == "Normal":
        return stats.norm.pdf(array, 0, 1)
    elif distType == "Cauchy":
        return stats.cauchy.pdf(array)
    elif distType == "Laplace":
        return stats.laplace.pdf(array, 0, 1 / 2 ** 0.5)
    elif distType == "Poisson":
        return stats.poisson.pmf(array, 10)
    elif distType == "Uniform":
        return stats.uniform.pdf(array, -np.sqrt(3), 2 * np.sqrt(3))
    return []

def getInterval(distType):
    if distType == "Poisson":
        return (6, 14, 1)
    else:
        return (-4, 4, 0.01)

def getXs(distType):
    N = [10, 50, 100]
    result = []
    start, end, step = getInterval(distType)
    x = np.arange(start, end, step)
    for size in N:
        incorrectX = getDistribution(distType, size)
        correctX = []
        for elem in incorrectX:
            if elem >= start and elem <= end:
                correctX.append(elem)
        result.append(correctX)
    return result, x, start, end

def task41():
    N = [20, 60, 100]
    for distribution in distType:
        array, x, start, end = getXs(distribution)
        index = 1
        figure, axes = plt.subplots(1, 3, figsize = (15,5))
        for elem in array:
            plt.subplot(1, 3, index)
            plt.title(distribution + ", n = " + str(N[index - 1]))
            if distribution == "Poisson" or distribution == "Uniform":
                plt.step(x, getCDF(distribution, x), color ="blue", label = "cdf")
            else:
                plt.plot(x, getCDF(distribution, x), color ="blue", label = "cdf")
            ar = np.linspace(start, end)
            ecdf = ECDF(elem)
            y = ecdf(ar)
            plt.step(ar, y, color ="black", label = "ecdf")
            plt.xlabel("x")
            plt.ylabel("(e)cdf")
            plt.legend(loc = "lower right")
            plt.subplots_adjust(wspace = 0.5)
            if not os.path.isdir("task41"):
                os.makedirs("task41")
            plt.savefig("task41/" + distribution + ".png")
            index += 1

def task42():
    N = [20, 60, 100]
    koef = [0.5, 1, 2]
    for distribution in distType:
        array, x, start, end = getXs(distribution)
        index = 1
        figure, axes = plt.subplots(1, 3, figsize = (15,5))
        for elem in array:
           headers = [r'$h = h_n/2$', r'$h = h_n$', r'$h = 2 * h_n$']
           figure, axes = plt.subplots(1, 3, figsize = (15,5))
           plt.subplots_adjust(wspace = 0.5)
           i = 0
           for k in koef:
               kde = stats.gaussian_kde(elem, bw_method = "silverman")
               hn = kde.factor
               figure.suptitle(distribution +", n =" + str(N[index - 1]))
               axes[i].plot(x, getPDF(distribution, x), color ="black", alpha = 0.5, label = "pdf")
               axes[i].set_title(headers[i])
               sns.kdeplot(elem, ax = axes[i], bw_adjust = hn * k, label= "kde", color = "blue")
               axes[i].set_xlabel('x')
               axes[i].set_ylabel('f(x)')
               axes[i].set_ylim([0, 1])
               axes[i].set_xlim([start, end])
               axes[i].legend()
               i = i + 1
               if not os.path.isdir("task42"):
                   os.makedirs("task42")
               plt.savefig("task42/" + distribution + "KDE" + str(N[index - 1]) + ".png")#save result
           index += 1


task42()
