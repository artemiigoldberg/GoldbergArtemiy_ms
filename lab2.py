import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tabulate
from matplotlib.patches import Ellipse
import statistics
import matplotlib.transforms as transforms
from scipy import stats as stats
import scipy.optimize as opt
from scipy.stats import laplace, uniform
import math
import os
from scipy.stats import chi2, t, norm, moment

sizes = [20, 60, 100]
correlation_coef = [0, 0.5, 0.9]
mean = [0, 0]
num = 1000

get_usual = lambda num, cov: stats.multivariate_normal.rvs(mean, cov, num)
get_mixed = lambda num, cov: 0.9 * stats.multivariate_normal.rvs(mean, [[1, 0.9], [0.9, 1]], num) \
             + 0.1 * stats.multivariate_normal.rvs(mean, [[10, -0.9], [-0.9, 10]], num)

def create_table(num, cov, fun):
    coefs = calculate_coefs(num, cov, fun)
    rows =[]
    rows.append(['$E(z)$', np.around(median(coefs['pearson']), decimals = 3),
                np.around(median(coefs['spearman']), decimals = 3),
                np.around(median(coefs['quadrat']), decimals = 3)])
    rows.append(['$E(z^2)$', np.around(quadric_median(coefs['pearson'], num), decimals = 3),
                np.around(quadric_median(coefs['spearman'], num), decimals = 3),
                np.around(quadric_median(coefs['quadrat'], num), decimals = 3)])
    rows.append(['$D(z)$', np.around(variance(coefs['pearson']), decimals = 3),
                np.around(variance(coefs['spearman']), decimals = 3),
                np.around(variance(coefs['quadrat']), decimals = 3)])
    return rows

def median(data):
    return np.median(data)

def quadric_median(data, num):
    return np.median([pow(data[k], 2) for k in range(num)])

def variance(data):
    return statistics.variance(data)

def calculate_coefs(num, cov, fun):
    coefs = {'pearson' : [], 'spearman' : [], 'quadrat' : []}
    for i in range(num):
        data = fun(num, cov)
        x, y = data[:, 0], data[:, 1]
        coefs['pearson'].append(stats.pearsonr(x, y)[0])
        coefs['spearman'].append(stats.spearmanr(x, y)[0])
        coefs['quadrat'].append(np.mean(np.sign(x - median(x)) * np.sign(y - median(y))))
    return coefs

def calculate_usual():
    for size in sizes:
        print("n = ", size)
        table = []
        for coef in correlation_coef:
            cov = [[1.0, coef], [coef, 1.0]]
            extension_table = create_table(size, cov, get_usual)
            title_row = ["$\\correlation_coef$ = " + str(coef), '$r$', '$r_S$', '$r_Q$']
            table.append([])
            table.append(title_row)
            table.extend(extension_table)
        print(tabulate.tabulate(table, headers=[]))

def calculate_mixed():
    table = []
    for size in sizes:
        extension_table = create_table(size, None, get_mixed)
        title_row = ["$n = " + str(size) + "$", '$r$', '$r_S$', '$r_Q$']
        table.append(title_row)
        table.extend(extension_table)
    print(tabulate.tabulate(table, headers=[]))

def create_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    r_x = np.sqrt(1 + pearson)
    r_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=r_x * 2, height=r_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_ellipses(samples):
    plt.rcParams['figure.figsize'] = [18, 12]
    num = len(samples[0])
    fig, ax = plt.subplots(1, len(samples))
    fig.suptitle("n = " + str(num))
    titles = ['$\\rho = 0$', '$\\rho = 0.5$', '$\\rho = 0.9$']
    i = 0
    for sample in samples:
        x = sample[:, 0]
        y = sample[:, 1]
        ax[i].scatter(x, y, c='red', s=6)
        create_ellipse(x, y, ax[i], edgecolor='gray')
        ax[i].scatter(np.mean(x), np.mean(y), c='blue', s=6)
        ax[i].set_title(titles[i])
        i += 1
    plt.savefig(
        "Ellipse n = " + str(num) + ".png",
        format='png'
    )

def ellipse_task():
    samples = []
    for num in sizes:
        for coef in correlation_coef:
            samples.append(get_usual(num, [[1.0, coef], [coef, 1.0]]))
        plot_ellipses(samples)
        samples = []

def task5():
    calculate_usual()
    calculate_mixed()
    ellipse_task()


points = 20
start, end = -1.8, 2
step = 0.2
mu, sigma_squared = 0, 1
perturbations = [10, -10]
coefs = 2, 2


def LST(x, y):
    b1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    b0 = np.mean(y) - b1 * np.mean(x)
    return b0, b1


def LMT(x, y, initial):
    fun = lambda beta: np.sum(np.abs(y - beta[0] - beta[1] * x))
    result = opt.minimize(fun, initial)
    b0 = result['x'][0]
    b1 = result['x'][1]
    return b0, b1


def find_all_coefs(x, y):
    bs0, bs1 = LST(x, y)
    bm0, bm1 = LMT(x, y, np.array([bs0, bs1]))
    return bs0, bs1, bm0, bm1


def print_results(all_coef):
    bs0, bs1, bm0, bm1 = all_coef
    print("Критерий наименьших квадратов")
    print('a_lst = ' + str(np.around(bs0, decimals=2)))
    print('b_lst = ' + str(np.around(bs1, decimals=2)))
    print("Критерий наименьших модулей")
    print('a_lmt = ' + str(np.around(bm0, decimals=2)))
    print('b_lmt = ' + str(np.around(bm1, decimals=2)))


def criteria_comparison(x, all_coef):
    a_lst, b_lst, a_lmt, b_lmt = all_coef
    model = lambda x: coefs[0] + coefs[1] * x
    lsc = lambda x: a_lst + b_lst * x
    lmc = lambda x: a_lmt + b_lmt * x

    sum_lst, sum_lmt = 0, 0
    for el in x:
        y_lst = lsc(el)
        y_lmt = lmc(el)
        y_model = model(el)
        sum_lst += pow(y_model - y_lst, 2)
        sum_lmt += pow(y_model - y_lmt, 2)

    if sum_lst < sum_lmt:
        print("LS wins - ", sum_lst, " < ", sum_lmt)
    else:
        print("LM wins - ", sum_lmt, " < ", sum_lst)


def plot_regression(x, y, type, estimates):
    a_ls, b_ls, a_lm, b_lm = estimates
    plt.scatter(x, y, label="Sample", edgecolor='gray', color='gray')
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Model', color='aqua')
    plt.plot(x, x * (b_ls * np.ones(len(x))) + a_ls * np.ones(len(x)), label='МНК', color='blue')
    plt.plot(x, x * (b_lm * np.ones(len(x))) + a_lm * np.ones(len(x)), label='МНМ', color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.savefig(type + '.png', format='png')
    plt.show()
    plt.close()


def task6():
    print("Without pertrubations\n")
    x = np.linspace(start, end, points)
    y = coefs[0] + coefs[1] * x + stats.norm(0, 1).rvs(points)
    all_coefs = find_all_coefs(x, y)
    print_results(all_coefs)
    criteria_comparison(x, all_coefs)
    plot_regression(x, y, "without", all_coefs)
    print("\n")
    print("With pertrubations\n")
    y[0] += perturbations[0]
    y[-1] += perturbations[1]
    all_coefs = find_all_coefs(x, y)
    print_results(all_coefs)
    criteria_comparison(x, all_coefs)
    plot_regression(x, y, "with", all_coefs)
    print("\n")


points = 20
sample_size_normal = 100
sample_size_laplace = 20
start, end = -1.4, 1.4
alpha = 0.05
p_ = 1 - alpha
mu_, sigma_squared = 0, 1


def find_k(size):
    return math.ceil(1.72 * (size) ** (1 / 3))


def max_likelihood_estimation(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    print("mu = ", np.around(mu, decimals=3),
          " sigma=", np.around(sigma, decimals=3))
    return mu, sigma


def calculate_chi2(p, n, sample_size):
    tmp = np.multiply((n - sample_size * p), (n - sample_size * p))
    chi2 = np.divide(tmp, p * sample_size)
    return chi2


def is_hypo_accepted(quantile, chi2):
    if quantile > np.sum(chi2):
        return True
    return False


def find_all_probabilities(borders, hypothesis, sample, k):
    p = np.array(hypothesis(start))
    n = np.array(len(sample[sample < start]))

    for i in range(k - 2):
        p_i = hypothesis(borders[i + 1]) - hypothesis(borders[i])
        p = np.append(p, p_i)
        n_i = len(sample[(sample < borders[i + 1]) & (sample >= borders[i])])
        n = np.append(n, n_i)

    p = np.append(p, 1 - hypothesis(end))
    n = np.append(n, len(sample[sample >= end]))

    return p, n


def chi_square_criterion(sample, mu, sigma, k):
    hypothesis = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
    borders = np.linspace(start, end, num=k - 1)
    p, n = find_all_probabilities(borders, hypothesis, sample, k)
    chi2 = calculate_chi2(p, n, len(sample))
    quantile = stats.chi2.ppf(p_, k - 1)
    isAccepted = is_hypo_accepted(quantile, chi2)
    return chi2, isAccepted, borders, p, n


def build_table(chi2, borders, p, n, sample_size):
    rows = []
    headers = ["$i$", "$\\Delta_i = [a_{i-1}, a_i)$", "$n_i$", "$p_i$",
               "$np_i$", "$n_i - np_i$", "$(n_i - np_i)^2/np_i$"]
    for i in range(0, len(n)):
        if i == 0:
            limits = ["$-\infty$", np.around(borders[0], decimals=3)]
        elif i == len(n) - 1:
            limits = [np.around(borders[-1], decimals=3), "$\infty$"]
        else:
            limits = [np.around(borders[i - 1], decimals=3), np.around(borders[i], decimals=3)]
        rows.append([i + 1, limits, n[i],
                     np.around(p[i], decimals=4),
                     np.around(p[i] * sample_size, decimals=3),
                     np.around(n[i] - sample_size * p[i], decimals=3),
                     np.around(chi2[i], decimals=3)])
    rows.append(["\\sum", "--", np.sum(n), np.around(np.sum(p), decimals=4),
                 np.around(np.sum(p * sample_size), decimals=3),
                 -np.around(np.sum(n - sample_size * p), decimals=3),
                 np.around(np.sum(chi2), decimals=3)]
                )
    return tabulate.tabulate(rows, headers)


def check_acception(isAccepted):
    if isAccepted:
        print("\nГипотезу принимаем")
    else:
        print("\nГипотезу принимаем!")


def calcucate_normal():
    k = find_k(sample_size_normal)
    normal_sample = np.random.normal(0, 1, sample_size_normal)
    mu, sigma = max_likelihood_estimation(normal_sample)
    chi2, isAccepted, borders, p, n = chi_square_criterion(normal_sample, mu, sigma, k)
    print(build_table(chi2, borders, p, n, 100))
    check_acception(isAccepted)


def calcucate_laplace():
    k = find_k(sample_size_laplace)
    laplace_sample = distribution = laplace.rvs(size=20, scale=1 / math.sqrt(2), loc=0)
    mu, sigma = max_likelihood_estimation(laplace_sample)
    chi2, isAccepted, borders, p, n = chi_square_criterion(laplace_sample, mu, sigma, k)
    print(build_table(chi2, borders, p, n, 20))
    check_acception(isAccepted)


def task7():
    calcucate_normal()
    calcucate_laplace()

gamma = 0.95
alpha = 0.05

def student_mo(samples, alpha):
    n = len(samples)
    q_1 = np.mean(samples) - np.std(samples) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    q_2 = np.mean(samples) + np.std(samples) * t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n - 1)
    return q_1, q_2

def chi_sigma(samples, alpha):
    n = len(samples)
    q_1 =  np.std(samples) * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    q_2 = np.std(samples) * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
    return q_1, q_2


def as_mo(samples, alpha):
    n = len(samples)
    q_1 = np.mean(samples) - np.std(samples) * norm.ppf(1 - alpha / 2) / np.sqrt(n)
    q_2 = np.mean(samples) + np.std(samples) * norm.ppf(1 - alpha / 2) / np.sqrt(n)
    return q_1, q_2


def as_sigma(samples, alpha):
    n = len(samples)
    s = np.std(samples)
    U = norm.ppf(1 - alpha / 2) * np.sqrt((moment(samples, 4) / (s * s * s * s) + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2


def task8():
  samples20 = np.random.normal(0, 1, size=20)
  samples100 = np.random.normal(0, 1, size=100)
  student_20 = student_mo(samples20, alpha)
  student_100 = student_mo(samples100, alpha)
  chi_20 = chi_sigma(samples20, alpha)
  chi_100 = chi_sigma(samples100, alpha)
  as_mo_20 = as_mo(samples20, alpha)
  as_mo_100 = as_mo(samples100, alpha)
  as_d_20 = as_sigma(samples20, alpha)
  as_d_100 = as_sigma(samples100, alpha)

  print(f"Classic:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(student_100) + " \t sigma: " + str(chi_100) + "\n")

  print(f"Asymptotic:\n"
        f"n = 20 \n"
        f"\t\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
        f"n = 100 \n"
        f"\t\t m: " + str(as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")

  # additional task - build histogram + intervals for classic
  figure, axes = plt.subplots(2, 2, figsize = (11.8, 3.9))#three different graphics in pic
  plt.subplots_adjust(wspace=0.5)#fix title landing
  figure.suptitle("Histograms and intervals for classic", y = 1, fontsize = 20)#name graphic
  axes[0][0].hist(samples20, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][0].set_title("N(0,1) histogram,n = 20")

  axes[0][1].hist(samples100, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][1].set_title("N(0,1) histogram, n = 100")

  axes[1][0].set_ylim(-0.1, 0.5)
  axes[1][0].plot(student_20, [0,0], 'ro-', label = 'm interval, n = 20')
  axes[1][0].plot(student_100, [0.1, 0.1], 'bo-', label = 'm interval, n = 100')
  axes[1][0].legend()
  axes[1][0].set_title('m intervals')

  axes[1][1].set_ylim(-0.1, 0.5)
  axes[1][1].plot(chi_20, [0,0], 'ro-', label = 'sigma interval, n = 20')
  axes[1][1].plot(chi_100, [0.1, 0.1], 'bo-', label='sigma interval, n = 100')
  axes[1][1].legend()
  axes[1][1].set_title("sigma intervals")

  plt.show()

  # additional task - build histogram + intervals
  figure, axes = plt.subplots(2, 2, figsize = (11.8, 3.9))#three different graphics in pic
  plt.subplots_adjust(wspace=0.5)#fix title landing
  figure.suptitle("Histograms and intervals for asymptotic", y = 1, fontsize = 20)#name graphic
  axes[0][0].hist(samples20, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][0].set_title("N(0,1) histogram,n = 20")

  axes[0][1].hist(samples100, density = 1, edgecolor = "blue", alpha = 0.3)
  axes[0][1].set_title("N(0,1) histogram, n = 100")

  axes[1][0].set_ylim(-0.1, 0.5)
  axes[1][0].plot(as_mo_20, [0,0], 'ro-', label = 'm interval, n = 20')
  axes[1][0].plot(as_mo_100, [0.1, 0.1], 'bo-', label = 'm interval, n = 100')
  axes[1][0].legend()
  axes[1][0].set_title('m intervals')

  axes[1][1].set_ylim(-0.1, 0.5)
  axes[1][1].plot(as_d_20, [0,0], 'ro-', label = 'sigma interval, n = 20')
  axes[1][1].plot(as_d_100, [0.1, 0.1], 'bo-', label='sigma interval, n = 100')
  axes[1][1].legend()
  axes[1][1].set_title("sigma intervals")

  plt.show()

task8()