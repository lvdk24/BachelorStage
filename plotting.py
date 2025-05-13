import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from TitanQ import base_path, calc_path, param_path, bonds_path, magn_filt_ratio
from main import getVarEngVal, nspins_ls, alpha_ls, timeout_ls, precision_ls, load_engVal, calcRelErr_vs_timeout, trainingLoop_TQ

plotting_diff_RBMEng = True
plotting_UF_varEng = False

def standardize(distribution):
    newDistribution = (distribution-np.mean(distribution))/np.std(distribution)
    return newDistribution

# def plotEngDist(nspins, alpha, bins, figCounter):
#     varEngValues = np.loadtxt(f"{base_path}/calculations/varEng/varEngValues/varEngValues_{nspins}_{alpha}.csv",delimiter=",")
#     hist, bins_hist = np.histogram()
#     plt.figure(figCounter)
#     plt.hist(varEngValues, bins = 20)
#     plt.xlabel("Variational Energy")
#     plt.ylabel("Amount")
#     plt.title(f"varEng distribution from initial TitanQ states, n={nspins}, alpha={alpha}")
#     plt.show()
#     plt.savefig(f"{base_path}/calculations/varEng/varEngPlots/varEngPlot_dist_{nspins}_{alpha}")
#
# def plotEngDist_diff(nspins, alpha, figCounter):
#
#     # setting paths
#     distributions_path = f"{base_path}/varParTitanQ/distributions"
#
#     # Load the distributions
#     distribution_path = f"{distributions_path}/{nspins}_{alpha}_Eloc.h5"
#
#     varEngValues = np.loadtxt(f"{base_path}/calculations/varEng/varEngValues/varEngValues_{nspins}_{alpha}.csv",delimiter=",")
#
#     with h5py.File(distribution_path, "r") as f:
#                 print("Keys: %s" % f.keys())
#
#                 Eloc = f['Eloc'][()]  # returns as a numpy array
#                 UltraFast_Values = f['RBMEnergy'][()]
#
#     # figure counter
#     plt.figure(figCounter)
#
#     # plotting histograms with standardization to compare
#     plt.hist(standardize(UltraFast_Values), alpha = 0.5, label = " UF")
#     plt.hist(standardize(varEngValues), alpha = 0.5, label = "TitanQ")
#
#     # aesthetics
#     plt.legend()
#     plt.xlabel("Variational Energy")
#     plt.ylabel("Amount")
#     plt.title(f"Distribution of varEng from TitanQ vs. UltraFast, n={nspins}, alpha={alpha}")
#
#     # saving & showing
#     plt.show()
#     plt.savefig(f"{base_path}/calculations/varEng/varEngComparison/varEngPlot_comp_{nspins}_{alpha}")

def make_RBMEng_diff_plot(nspins, alpha, timeout, nruns, precision_param):
    figcount = 1
    # getting histogram values from TitanQ
    RBMEngVal_TQ, RBMEngVal_UF, _, _, _ = load_engVal(nspins, alpha, timeout, nruns, precision_param)

    hist_RBMEng_TQ, bins_RBMEng_TQ = np.histogram(RBMEngVal_TQ, bins=60, density=True)
    hist_RBMEng_TQ = hist_RBMEng_TQ / np.sum(hist_RBMEng_TQ)

    # getting histogram values from UltraFast
    hist_RBMEng_UF, binsUF = np.histogram(-RBMEngVal_UF, bins=bins_RBMEng_TQ, density=True)
    hist_RBMEng_UF = hist_RBMEng_UF / np.sum(hist_RBMEng_UF)

    # starting figure
    plt.figure(figcount)
    # plt.bar(bins_var_TQ[:-1], hist_RBMEng_TQ, width=np.diff(bins_var_TQ), alpha=0.3, color = 'cyan', edgecolor="black",label="RBMEngTQ")
    plt.step((bins_RBMEng_TQ[:-1] + bins_RBMEng_TQ[1:]) / 2, hist_RBMEng_TQ, color='blue', label="TQ")
    plt.bar(bins_RBMEng_TQ[:-1], hist_RBMEng_UF, width=np.diff(bins_RBMEng_TQ), alpha=0.6, color='yellow',
            edgecolor="black", label="UF")

    # aesthetics
    plt.xlabel("RBM Energy")
    plt.ylabel("Probability")
    myTitle = f"Difference in RBM Energy TQ vs UF, n={nspins}, " + r"$\alpha$" +f"={alpha}, " + r"$\tau$" +f"={timeout}s, precision={precision_param}"
    plt.title(myTitle, loc='center', wrap=True)
    plt.legend(loc="upper right")

    plt.savefig(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEngPlots/RBMEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

def make_RBMEng_diff_prec_plot(nspins, alpha, timeout, nruns, precision_ls):
    """
    This functions also plots the RBM energy distribution but now with two kinds of precision from TitanQ
    :param nspins:
    :param alpha:
    :param timeout:
    :param nruns:
    :param precision_ls:
    :return:
    """
    figcount = 1
    # getting histogram values from TitanQ

    RBMEngVal_TQ_standard, RBMEngVal_UF, _, _, _ = load_engVal(nspins, alpha, timeout, nruns, precision_ls[0])
    RBMEngVal_TQ_high, _, _, _, _ = load_engVal(nspins, alpha, timeout, nruns, precision_ls[1])

    #TQ standard precision
    hist_RBMEng_TQ_standard, bins_RBMEng_TQ = np.histogram(RBMEngVal_TQ_standard, bins=60, density=True)
    hist_RBMEng_TQ_standard = hist_RBMEng_TQ_standard / np.sum(hist_RBMEng_TQ_standard)

    #TQ high precision
    hist_RBMEng_TQ_high, bins_RBMEng_TQ_high = np.histogram(RBMEngVal_TQ_high, bins=bins_RBMEng_TQ, density=True)
    hist_RBMEng_TQ_high = hist_RBMEng_TQ_high / np.sum(hist_RBMEng_TQ_high)

    # getting histogram values from UltraFast
    hist_RBMEng_UF, binsUF = np.histogram(-RBMEngVal_UF, bins=bins_RBMEng_TQ, density=True)
    hist_RBMEng_UF = hist_RBMEng_UF / np.sum(hist_RBMEng_UF)

    # starting figure
    plt.figure(figcount)
    # plt.bar(bins_var_TQ[:-1], hist_RBMEng_TQ, width=np.diff(bins_var_TQ), alpha=0.3, color = 'cyan', edgecolor="black",label="RBMEngTQ")
    plt.step((bins_RBMEng_TQ[:-1] + bins_RBMEng_TQ[1:]) / 2, hist_RBMEng_TQ_standard, color='blue', label="TQ_st")
    plt.step((bins_RBMEng_TQ_high[:-1] + bins_RBMEng_TQ_high[1:]) / 2, hist_RBMEng_TQ_high, color='red', label="TQ_hi")
    plt.bar(bins_RBMEng_TQ[:-1], hist_RBMEng_UF, width=np.diff(bins_RBMEng_TQ), alpha=0.6, color='yellow',
            edgecolor="black", label="UF")

    # aesthetics
    plt.xlabel("RBM Energy")
    plt.ylabel("Probability")
    myTitle = f"Difference in RBM Energy TQ vs UF, n={nspins}, " + r"$\alpha$" + f"={alpha}, " + r"$\tau$" + f"={timeout}s"
    plt.title(myTitle, loc='center', wrap=True)
    plt.legend(loc="upper right")

    plt.savefig(f"{calc_path}/RBMEng/RBMEngPlots_comp_prec/RBMEngPlot_comp_prec_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

def make_varEng_diff_plot(nspins, alpha, timeout, nruns, precision_param):
    figcount = 1
    # getting histogram values from TitanQ
    _, _, _, varEngVal_UF, varEngVal_TQ = load_engVal(nspins, alpha, timeout, nruns, precision_param)

    hist_varEng_TQ, bins_varEng_TQ = np.histogram(varEngVal_TQ, bins=60, density=True)
    hist_varEng_TQ = hist_varEng_TQ / np.sum(hist_varEng_TQ)

    # getting histogram values from UltraFast
    hist_varEng_UF, binsUF = np.histogram(varEngVal_UF, bins=bins_varEng_TQ, density=True)
    hist_varEng_UF = hist_varEng_UF / np.sum(hist_varEng_UF)

    # starting figure
    plt.figure(figcount)
    # plt.bar(bins_var_TQ[:-1], hist_RBMEng_TQ, width=np.diff(bins_var_TQ), alpha=0.3, color = 'cyan', edgecolor="black",label="RBMEngTQ")
    plt.step((bins_varEng_TQ[:-1] + bins_varEng_TQ[1:]) / 2, hist_varEng_TQ, color='blue', label="TQ")
    plt.bar(bins_varEng_TQ[:-1], hist_varEng_UF, width=np.diff(bins_varEng_TQ), alpha=0.6, color='yellow',
            edgecolor="black", label="UF")

    # aesthetics
    myTitle = f"Difference in variational Energy TQ vs UF, n={nspins}, " + r"$\alpha$" +f"={alpha}, " + r"$\tau$" + f"={timeout}s, nruns={nruns}, precision={precision_param}"
    plt.xlabel("Variational Energy")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.title(myTitle, loc='center', wrap=True)
    plt.savefig(f"{calc_path}/varEng/precision_{precision_param}/varEngPlots/varEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

def makePlot_varEng_differentCalculation(nspins, alpha, timeout, nruns, precision_param):
    figcount = 1
    # getting histogram values from TitanQ
    _, _, _, _, varEngVal_TQ = load_engVal(nspins, alpha, timeout, nruns, precision_param)

    hist_varEng_TQ, bins_varEng_TQ = np.histogram(varEngVal_TQ, bins=60, density=True)
    hist_varEng_TQ = hist_varEng_TQ / np.sum(hist_varEng_TQ)

    varEng_new = np.loadtxt(f"{calc_path}/varEng/test_varEng_calc/varEng_{nspins}_{alpha}_{timeout}_{nruns}.csv", delimiter=",")

    hist_varEng_new, bins_new = np.histogram(varEng_new, bins=60, density=True)
    hist_varEng_new = hist_varEng_new / np.sum(hist_varEng_new)

    # starting figure
    plt.figure(figcount)
    # plt.bar(bins_var_TQ[:-1], hist_RBMEng_TQ, width=np.diff(bins_var_TQ), alpha=0.3, color = 'cyan', edgecolor="black",label="RBMEngTQ")
    plt.step((bins_varEng_TQ[:-1] + bins_varEng_TQ[1:]) / 2, hist_varEng_TQ, color='blue', label="old")
    plt.bar(bins_varEng_TQ[:-1], hist_varEng_new, width=np.diff(bins_varEng_TQ), alpha=0.6, color='yellow',
            edgecolor="black", label="new")

    # aesthetics
    myTitle = f"Difference in variational Energy TQ, two calculation methods, n={nspins}, " + r"$\alpha$" +f"={alpha}, " + r"$\tau$" + f"={timeout}s, nruns={nruns}, precision={precision_param}"
    plt.xlabel("Variational Energy")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.title(myTitle, loc='center', wrap=True)
    # plt.savefig(f"{calc_path}/varEng/precision_{precision_param}/varEngPlots/varEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

# makePlot_varEng_differentCalculation(36,2,2,32,'high')

def make_varEng_diff_prec_plot(nspins, alpha, timeout, nruns, precision_ls):
    figcount = 1
    # getting histogram values from TitanQ
    _, _, _, varEngVal_UF, varEngVal_TQ_standard = load_engVal(nspins, alpha, timeout, nruns, precision_ls[0])
    _, _, _, _, varEngVal_TQ_high = load_engVal(nspins, alpha, timeout, nruns, precision_ls[1])

    #hist standard precision
    hist_varEng_TQ_standard, bins_varEng_TQ = np.histogram(varEngVal_TQ_standard, bins=60, density=True)
    hist_varEng_TQ_standard = hist_varEng_TQ_standard / np.sum(hist_varEng_TQ_standard)

    # hist high precision
    hist_varEng_TQ_high, bins_varEng_TQ_high = np.histogram(varEngVal_TQ_high, bins=bins_varEng_TQ, density=True)
    hist_varEng_TQ_high = hist_varEng_TQ_high / np.sum(hist_varEng_TQ_high)

    # getting histogram values from UltraFast
    hist_varEng_UF, binsUF = np.histogram(varEngVal_UF, bins=bins_varEng_TQ, density=True)
    hist_varEng_UF = hist_varEng_UF / np.sum(hist_varEng_UF)

    # starting figure
    plt.figure(figcount)
    # plt.bar(bins_var_TQ[:-1], hist_RBMEng_TQ, width=np.diff(bins_var_TQ), alpha=0.3, color = 'cyan', edgecolor="black",label="RBMEngTQ")
    plt.step((bins_varEng_TQ[:-1] + bins_varEng_TQ[1:]) / 2, hist_varEng_TQ_standard, color='blue', label="TQ_st")
    plt.step((bins_varEng_TQ_high[:-1] + bins_varEng_TQ_high[1:]) / 2, hist_varEng_TQ_high, color='red', label="TQ_hi")

    plt.bar(bins_varEng_TQ[:-1], hist_varEng_UF, width=np.diff(bins_varEng_TQ), alpha=0.6, color='yellow',
            edgecolor="black", label="UF")

    # aesthetics
    myTitle = f"Difference in variational Energy TQ vs UF, n={nspins}, " + r"$\alpha$" + f"={alpha}, " + r"$\tau$" + f"={timeout}s, nruns={nruns}"
    plt.xlabel("Variational Energy")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.title(myTitle, loc='center', wrap=True)
    plt.savefig(f"{calc_path}/varEng/varEngPlots_comp_prec/varEngPlots_comp_prec_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

def make_relErr_vs_nspins_plot(nspins_ls,alpha_ls):

    plt.figure()
    col_ind=0
    col=['orange', 'magenta']
    for alpha in alpha_ls:
        relErr_arr = np.loadtxt(f"{base_path}/calculations/accuracy/relErr_vs_nspins/relErr_{alpha}_10_8.csv", delimiter = ",")
        plt.plot(nspins_ls, relErr_arr, color = col[col_ind], label=r"$\alpha$" + f"={alpha}")
        col_ind += 1

    #aesthetics
    plt.xticks(nspins_ls)
    plt.xlabel("nspins")
    plt.ylabel("Relative error")
    plt.title("Relative error, nruns=8, " + r"$\tau$" + "=10s")
    plt.legend()

    #saving and showing the figure
    plt.savefig(f"{calc_path}/accuracy/relErr_vs_nspins.png",bbox_inches='tight')
    plt.show()

def make_relErr_vs_timeout_plot(nspins_ls, alpha, nruns):
    figcount = 1
    plt.figure(figcount)

    #creating gradient in colours
    def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    #defining the two colours
    c1 = 'red'  # blue
    c2 = 'yellow'  # green

    nspins_counter = 1
    for nspins_ind in nspins_ls:

        #Hier moet nog de mean bij +stddev/errorbars van de split e
        relErr_arr = np.loadtxt(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout/relErr_{nspins_ind}_{alpha}_{nruns}.csv", delimiter = ",")
        plt.plot(timeout_ls, relErr_arr, color=colorFader(c1, c2, nspins_counter / len(nspins_ls)), label=f'n={nspins_ind}')
        nspins_counter += 1

    # aesthetics
    plt.xticks(timeout_ls)
    plt.xlabel("timeout (s)")
    plt.ylabel("Relative error")
    plt.title(r"$\alpha$" + f"={alpha}, nruns={nruns}")
    plt.legend()

    # saving and showing the figure
    plt.savefig(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout_{alpha}_{nruns}.png", bbox_inches='tight')
    figcount += 1
    plt.show()

def makePlot_relErr_vs_timeout_split_states(nspins_ls, alpha, timeout_ls,nruns, split_bins = 4):
    figcount = 1
    plt.figure(figcount)

    #creating gradient in colours
    def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    #defining the two colours
    c1 = 'red'  # blue
    c2 = 'yellow'  # green

    # relErr_arr = []
    nspins_counter = 1

    for nspins_ind in nspins_ls:
        # for split_ind in range(split_bins):
        relErr_arr = calcRelErr_vs_timeout(nspins_ind, alpha, timeout_ls, 32, 'high', True, 4)
            #Hier moet nog de mean bij +stddev/errorbars van de split e
            # relErr_arr.append(np.loadtxt(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout/split_states/relErr_{nspins_ind}_{alpha}_{nruns}_{split_ind + 1}of{split_bins}.csv", delimiter = ","))

        avg_relErr_arr = np.mean(relErr_arr, axis = 0)
        stdev_relErr_arr = np.std(relErr_arr, axis = 0)
        plt.errorbar(timeout_ls, avg_relErr_arr, yerr = stdev_relErr_arr, capsize = 6, color=colorFader(c1, c2, nspins_counter / len(nspins_ls)), label=f'n = {nspins_ind}')
        nspins_counter += 1

    # aesthetics
    # plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.1,2,4,10,16, 24])
    plt.xlabel("timeout (s)")
    plt.ylabel("Relative error")
    plt.title(r"$\alpha$" + f"={alpha}, runs = {nruns}")
    plt.legend()

    # saving and showing the figure
    # plt.savefig(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout_split_{alpha}_{nruns}_lower_timeout_values.png", bbox_inches='tight')
    figcount += 1
    plt.show()

def makePlot_magn_filt_ratio(nspins_ls, alpha, timeout_ls, nruns, precision_param):
    figcount = 1
    plt.figure(figcount)




    ratio_arr = np.loadtxt(f"{calc_path}/accuracy/precision_{precision_param}/magn_filt_ratio_{alpha}_{nruns}.csv",delimiter=",")

    for ratio_ind in range(len(ratio_arr)):

        plt.plot(nspins_ls, ratio_arr[ratio_ind], label = f"t={timeout_ls[ratio_ind]}")

    plt.xticks(nspins_ls)
    plt.xlabel("nspins")
    plt.ylabel("Magnetic filter ratio")
    plt.legend()
    plt.title(f"Magnetic Filter ratio vs nspins, " + r"$\alpha$" + f"={alpha}")
    plt.savefig(f"{calc_path}/accuracy/precision_high/magn_filt_ratio_{alpha}_{nruns}.png", bbox_inches='tight')
    figcount += 1
    plt.show()

def makePlot_training_varEng(nspins, alpha):
    varEngVal, varEngVal_arr, _, _, epoch = trainingLoop_TQ(nspins, alpha)

    x_val = np.arange(epoch)

    plt.figure()
    plt.plot(x_val, varEngVal_arr)
    plt.axhline(varEngVal)

    plt.show()

# makePlot_training_varEng(16, 2)

# def UF_varEng_dist_comp(nspins, alpha, timeout, nruns, precision_param):
#
#     # loading energy values
#     _,_,_,varEngVal_UF,_ = load_engVal(nspins, alpha, timeout, nruns, precision_param)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#
#     def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
#         c1 = np.array(mpl.colors.to_rgb(c1))
#         c2 = np.array(mpl.colors.to_rgb(c2))
#         return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
#
#     # defining the two colours
#     c1 = 'red'  # blue
#     c2 = 'yellow'  # green
#
#
#     # colors = ['r', 'g', 'b', 'y']
#     yticks = np.linspace(1,32,32)
#     for eng_ind in range(len(yticks)):
#         # Generate the random data for the y=k 'layer'.
#         xs = varEngVal_UF[eng_ind]
#         # ys = np.random.rand(20)
#         # You can provide either a single color or an array with the same length as
#         # xs and ys. To demonstrate this, we color the first bar of each set cyan.
#         # cs = [c] * len(xs)
#         # cs[0] = 'c'
#
#         # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
#         # ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)
#         hist_varEng_UF, binsUF = np.histogram(varEngVal_UF, bins=60, density=True)
#         hist_varEng_UF = hist_varEng_UF / np.sum(hist_varEng_UF)
#         plt.bar(binsUF[:-1], hist_varEng_UF, width=np.diff(binsUF), alpha=0.6, color=colorFader(c1, c2, eng_ind / len(varEngVal_UF)),
#                 edgecolor="black", label="UF")
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # On the y-axis let's only label the discrete values that we have data for.
#     ax.set_yticks(yticks)
#
#     plt.show()
#


# Fixing random state for reproducibility

# def UF_varEng_dist_comp(nspins, alpha, timeout, nruns, precision_param):
#     # np.random.seed(19680801)
#     figcount = 1
#     # loading energy values
#     _,_,_,varEngVal_UF,_ = load_engVal(nspins, alpha, timeout, nruns, precision_param)
#     varEng_1 = varEngVal_UF[0]
#     varEng_2 = varEngVal_UF[1]
#
#     fig = plt.figure(figcount)
#     ax = fig.add_subplot(projection='3d')
#
#     colors = ['r']#, 'g', 'b', 'y']
#     yticks = [1, 2]#, 1, 0]
#     for c, k in zip(colors, yticks):
#         # Generate the random data for the y=k 'layer'.
#         # xs = np.arange(20)
#         # ys = np.random.rand(20)
#         hist_varEng_UF, binsUF = np.histogram(varEngVal_UF, bins=60, density=True)
#         hist_varEng_UF = hist_varEng_UF / np.sum(hist_varEng_UF)
#
#         # starting figure
#         # plt.figure(figcount)
#
#         plt.bar(binsUF[:-1], hist_varEng_UF, width=np.diff(binsUF), alpha=0.6, color='yellow',edgecolor="black", label="UF")
#
#
#         # You can provide either a single color or an array with the same length as
#         # xs and ys. To demonstrate this, we color the first bar of each set cyan.
#         cs = [c] * len(hist_varEng_UF)
#         cs[0] = 'c'
#
#         # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
#         # plt.bar(hist_varEng_UF, binsUF[:-1], zs=k, zdir='y', color=cs, alpha=0.8)
#         figcount += 1
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # On the y-axis let's only label the discrete values that we have data for.
#     ax.set_yticks(yticks)
#
#     plt.show()


# UF_varEng_dist_comp(16,2,10,32,'high')





# making distributions differences plots

# plotting relative error vs UF (with errorbars)
# makePlot_relErr_vs_timeout_split_states(nspins_ls, 2, timeout_ls, 32, split_bins=4)

# plotting magn filter ratio vs nspins
# makePlot_magn_filt_ratio(nspins_ls, 2, timeout_ls, 32, 'high')


# for alpha in alpha_ls:
#     make_relErr_vs_timeout_plot(nspins_ls, alpha, nruns=32)
        # for nspins in nspins_ls:
        # make_relErr_vs_timeout_plot(nspins_ls, alpha, 8)
        #     make_RBMEng_diff_plot(nspins, alpha, 10, 8, 'standard')
        #     make_varEng_diff_plot(nspins, alpha, 10, 8, 'standard')
        #     make_varEng_diff_prec_plot(nspins, alpha, 10, 8, precision_ls)
            # make_RBMEng_diff_plot(nspins,alpha,10,8)
    # make_RBMEng_diff_plot(64, 2, 60, 8)
    # make_RBMEng_diff_plot(64, 2, 10, 32)
    # make_varEng_diff_plot(36, 2, 10, 32)
# make_RBMEng_diff_plot(16,2,10,8,'high')
