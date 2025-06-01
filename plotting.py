import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import glob
from tqdm import tqdm
import json

from TitanQ import base_path, calc_path, param_path, bonds_path, magn_filt_ratio
from main import getVarEngVal, nspins_ls, alpha_ls, timeout_ls, precision_ls, load_engVal, calcRelErr_vs_timeout, trainingLoop_TQ, storeVal_path

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
    plt.savefig(f"{calc_path}/varEng/precision_{precision_param}/varEngPlots/varEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

# makePlot_varEng_differentCalculation(16,2,2,32,'high')

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
    c1 = '#182D66'
    c2 = '#BEF9FA'

    # relErr_arr = []
    nspins_counter = 1

    for nspins_ind in nspins_ls:
        # for split_ind in range(split_bins):
        relErr_arr = calcRelErr_vs_timeout(nspins_ind, alpha, timeout_ls, 32, 'high', True, 4)
            #Hier moet nog de mean bij +stddev/errorbars van de split e
            # relErr_arr.append(np.loadtxt(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout/split_states/relErr_{nspins_ind}_{alpha}_{nruns}_{split_ind + 1}of{split_bins}.csv", delimiter = ","))

        avg_relErr_arr = np.mean(relErr_arr, axis = 0)
        stdev_relErr_arr = np.std(relErr_arr, axis = 0)
        plt.errorbar(timeout_ls, avg_relErr_arr, yerr = stdev_relErr_arr, capsize = 4, color=colorFader(c1, c2, nspins_counter / len(nspins_ls)), label=f'n={nspins_ind}')
        nspins_counter += 1

    # aesthetics
    # plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.1,2,4,10,16, 24])
    plt.xlabel("timeout (s)")
    plt.ylabel("Relative error")
    plt.title(r"$\alpha$" + f"={alpha}, runs={nruns}")
    plt.legend()

    # saving and showing the figure
    plt.savefig(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout_split_{alpha}_{nruns}_lower_timeout_values.png", bbox_inches='tight')
    figcount += 1
    plt.show()

def makePlot_magn_filt_ratio(nspins_ls, alpha, timeout_ls, nruns, precision_param):
    figcount = 1
    plt.figure(figcount)

    def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    #defining the two colours
    c1 = '#182D66'
    c2 = '#BEF9FA'

    ratio_arr = np.loadtxt(f"{calc_path}/accuracy/precision_{precision_param}/magn_filt_ratio_{alpha}_{nruns}.csv",delimiter=",")
    nspins_counter = 1
    for ratio_ind in range(len(ratio_arr)):

        plt.plot(nspins_ls, ratio_arr[ratio_ind], color=colorFader(c1, c2, nspins_counter / len(nspins_ls)), label = f"t={timeout_ls[ratio_ind]}")
        nspins_counter += 1
    plt.xticks(nspins_ls)
    plt.xlabel("nspins")
    plt.ylabel("Magnetic filter ratio")
    plt.legend()
    plt.title(f"Magnetic Filter ratio vs nspins, " + r"$\alpha$" + f"={alpha}")
    plt.savefig(f"{calc_path}/accuracy/precision_high/magn_filt_ratio_{alpha}_{nruns}.png", bbox_inches='tight')
    figcount += 1
    plt.show()

# makePlot_magn_filt_ratio(nspins_ls, 4, timeout_ls, 32, 'high')

# def makePlot_hist_training_varEng(nspins, alpha, epochs):
#     # check how many files of this configuration exist already
#     list_of_files = glob.glob(f"{calc_path}/varEng/varEng_training_evolution/{nspins}_{alpha}_{epochs}/*.csv")
#
#     epoch_runs = len(list_of_files)
#
#     # varEngVal, varEngVal_arr, _, _, epoch = trainingLoop_TQ(nspins, alpha)
#
#     # x_val = np.arange(epochs)
#
#     for epoch_ind in range(epoch_runs):
#         varEngVal_arr = np.loadtxt(f"{calc_path}/varEng/varEng_training_evolution/{nspins}_{alpha}_{epochs}/varEng_evolution_{nspins}_{alpha}_{epoch_ind+1}of{epochs}.csv", delimiter=",")
#         figcount = 1
#         plt.figure(figcount)
#
#         hist_varEng_Evo_TQ, bins_varEng_Evo_TQ = np.histogram(varEngVal_arr, bins=60, density=True)
#         hist_varEng_Evo_TQ = hist_varEng_Evo_TQ / np.sum(hist_varEng_Evo_TQ)
#         # plt.axhline(varEngVal)
#
#         plt.step((bins_varEng_Evo_TQ[:-1] + bins_varEng_Evo_TQ[1:]) / 2, hist_varEng_Evo_TQ, color='blue',)
#
#         myTitle = f"Variational energy, epoch = {epoch_ind + 1}"
#         plt.xlabel("Variational Energy")
#         plt.ylabel("Probability")
#         # plt.legend(loc="upper right")
#         plt.title(myTitle, loc='center', wrap=True)
#         plt.savefig(f"{calc_path}/varEng/varEng_training_evolution/plots/varEng_Evo_{epoch_ind + 1}of{epochs}.png")
#         figcount += 1
#
#         plt.show()

def makePlot_training_varEngVal(nspins, alpha, epochs):

    with open(f"{storeVal_path}/varEng_evolution_{nspins}_{alpha}_{epochs}.json", 'r') as file:
        varEngVal_evolution = json.load(file)
        varEngVal_arr = varEngVal_evolution['varEngVal_arr']
        filt_samp_arr = varEngVal_evolution['varEngVal_evolution']



    plt.figure()
    x = np.arange(epochs)

    plt.plot(x, varEngVal_arr, label = "TQ training")
    plt.plot(x, filt_samp_arr, label = "filt ratio")
    plt.axhline(-0.701777, color = 'r',label = "QMC_eng")
    myTitle = f"Variational energy, n={nspins}, "+ r"$\alpha$" +f"={alpha}, " + f"epochs={epochs}"#, time/sweep={int(time_per_sweep)}s"
    plt.xlabel("epoch")
    plt.ylabel("Variational Energy")
    plt.legend()
    # plt.ylim(-0.75,-0.50)
    # plt.legend(loc="upper right")
    plt.title(myTitle, loc='center', wrap=True)
    # plt.savefig(f"{calc_path}/varEng/varEng_training_evolution/plots/varEngVal_Evo_{nspins}_{alpha}_{epochs}.png")

    plt.show()

def makePlot_training_varEngVal_copy(nspins, alpha, epochs):

    with open(f"{storeVal_path}/varEng_evolution_{nspins}_{alpha}_{epochs}.json", 'r') as file:
        varEngVal_evolution = json.load(file)
        varEngVal_arr = varEngVal_evolution['varEngVal_arr']
        filt_samp_arr = varEngVal_evolution['amount_of_filt_samples']

    varEng_col = "#69b3a2"
    filt_col = "#3399e6"


    # plt.figure()
    x = np.arange(epochs)
    #
    # plt.plot(x, varEngVal_arr, label = "TQ training")
    # plt.plot(x, filt_samp_arr, label = "filt ratio")
    # plt.axhline(-0.701777, color = 'r',label = "QMC_eng")
    # myTitle = f"Variational energy, n={nspins}, "+ r"$\alpha$" +f"={alpha}, " + f"epochs={epochs}"#, time/sweep={int(time_per_sweep)}s"
    # plt.xlabel("epoch")
    # plt.ylabel("Variational Energy")
    # plt.legend()
    # # plt.ylim(-0.75,-0.50)
    # # plt.legend(loc="upper right")
    # plt.title(myTitle, loc='center', wrap=True)
    # # plt.savefig(f"{calc_path}/varEng/varEng_training_evolution/plots/varEngVal_Evo_{nspins}_{alpha}_{epochs}.png")
    #
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    ax1.plot(x, varEngVal_arr, color=varEng_col, lw=3)
    ax2.plot(x, filt_samp_arr, color=filt_col, lw=4)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("variational Energy", color=varEng_col, fontsize=14)
    ax1.tick_params(axis="y", labelcolor=varEng_col)

    ax2.set_ylabel("Amount of filtered samples", color=filt_col, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=filt_col)

    fig.suptitle("VarEng & filt samples vs epochs", fontsize=20)
    fig.autofmt_xdate()

    plt.savefig(f"{calc_path}/varEng/varEng_training_evolution/plots/varEngVal_Evo_and_filtSamps_{nspins}_{alpha}_{epochs}.png")

    plt.show()

QMC_eng = [-0.701777,-0.678873,-0.673487 ]

def makePlot_locEng_speedup(timeout, nruns, precision_param):

    # initializing for alpha = 2
    num_filt_samples_arr = [8984,5986,4326,3389,2517,2022,1969] # alpha = 2

    runtime_new_avg = np.zeros((len(nspins_ls)))
    runtime_new_std = np.zeros((len(nspins_ls)))
    runtime_avg = np.zeros((len(nspins_ls)))
    runtime_std = np.zeros((len(nspins_ls)))

    # initializing for alpha = 4
    num_filt_samples_arr_4 = [11090, 10079, 7829, 5348, 3566, 2748, 3010]  # alpha = 4
    runtime_new_avg_4 = np.zeros((len(nspins_ls)))
    runtime_new_std_4 = np.zeros((len(nspins_ls)))
    runtime_avg_4 = np.zeros((len(nspins_ls)))
    runtime_std_4 = np.zeros((len(nspins_ls)))

    c1 = '#182D66'
    c2 = '#B1DAE3'

    for i in range(len(nspins_ls)):

        # for alpha = 2
        time_per_nspin_new = np.zeros((4))
        time_per_nspin = np.zeros((4))
        for run_ind in range(4):
            with open(f"{calc_path}/varEng/precision_{precision_param}/varEng_new_{nspins_ls[i]}_2_{timeout}_{nruns}_{run_ind+2}of5.json", 'r') as file:
                data_new = json.load(file)

            time_per_nspin_new[run_ind] = (data_new['runtime'] / num_filt_samples_arr[i])

            with open(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins_ls[i]}_2_{timeout}_{nruns}_{run_ind+2}of5.json", 'r') as file:
                data = json.load(file)
            time_per_nspin[run_ind]=(data['runtime'] / num_filt_samples_arr[i])

        runtime_new_avg[i] = np.mean(time_per_nspin_new)
        runtime_new_std[i] = np.std(time_per_nspin_new)
        runtime_avg[i] = np.mean(time_per_nspin)
        runtime_std[i] = np.std(time_per_nspin)

        # for alpha = 4
        time_per_nspin_new_4 = np.zeros((4))
        time_per_nspin_4 = np.zeros((4))

        for run_ind in range(4):
            with open(f"{calc_path}/varEng/precision_{precision_param}/varEng_new_{nspins_ls[i]}_4_{timeout}_{nruns}_{run_ind+1}of4.json", 'r') as file:
                data_new = json.load(file)

            time_per_nspin_new_4[run_ind] = (data_new['runtime'] / num_filt_samples_arr_4[i])

            with open(f"{calc_path}/varEng/precision_{precision_param}/varEng_{nspins_ls[i]}_4_{timeout}_{nruns}_{run_ind+1}of4.json", 'r') as file:
                data = json.load(file)
            time_per_nspin_4[run_ind]=(data['runtime'] / num_filt_samples_arr_4[i])

        runtime_new_avg_4[i] = np.mean(time_per_nspin_new_4)
        runtime_new_std_4[i] = np.std(time_per_nspin_new_4)
        runtime_avg_4[i] = np.mean(time_per_nspin_4)
        runtime_std_4[i] = np.std(time_per_nspin_4)

    x = nspins_ls

    plt.figure()
    plt.errorbar(x, runtime_avg, yerr=runtime_std, capsize = 4,color = c1, linestyle = 'dotted', label="Old, " + r"$\alpha$" + f"={2}")
    plt.errorbar(x, runtime_new_avg, yerr=runtime_new_std, capsize = 4, color = c2, linestyle='dotted', label="New (look-up table), " + r"$\alpha$" + f"={2}")
    plt.errorbar(x, runtime_avg_4, yerr=runtime_std_4, alpha = 0.8, capsize=4, color=c1, lw = 0.8, label="Old, " + r"$\alpha$" + f"={4}")
    plt.errorbar(x, runtime_new_avg_4, yerr=runtime_new_std_4, alpha = 0.8, capsize=4, color=c2, lw = 0.8, label="New (look-up table), " + r"$\alpha$" + f"={4}")
    plt.yscale('log')
    plt.xticks(x)
    plt.xlabel("nspins")
    plt.ylabel("Computation time per filtered sample (s)")
    plt.title("Computational advantage of using lookup table")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.savefig(f"{calc_path}/varEng/locEng_calc_methods_comp.png")
    plt.show()

# makePlot_locEng_speedup(2,32,'high')

def makePlot_sketch_timeout():

    x=np.arange(0,10,0.01)
    y=np.exp(-(x-2))+0.5
    y_2 = 0.5

    c1 = '#182D66'

    plt.figure()
    plt.plot(x, y, color = c1, label="TitanQ")
    plt.axhline(0.5, color = c1, linestyle = 'dashed', alpha = 0.5, label ="balancing line")
    plt.xlabel("Time / samples")
    plt.ylabel("Energy of state")
    plt.title("Sketch of thermalisation of TitanQ")
    plt.legend()
    plt.gca().axes.get_yaxis().set_ticklabels([])
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.savefig(f"{calc_path}/Thermalisation_sketch.png")
    plt.show()

def makePlot_timeProjection(timeout):
    # data = np.loadtxt(f"{base_path}/projections-MH-Ising-FPGA-Conservative-Optimistic.csv", delimiter = ",", skiprows=8)
    x = [16,36,64,100,144,196,256,324,400,484]
    x_2 = nspins_ls
    UF_arr =[1.4949295699999999e-05, 7.1075944e-05,0.00022066767475,0.0005290877813499999,0.0010919345298999999,0.0019848194393499996,0.0034447283587,0.00549388198105,0.00833878898185,0.0122014823203]
    UF_std_arr = [1.281692194360984e-07,2.400770237115583e-07,6.494463287045599e-07,1.4500511100564922e-06,4.320382517891083e-06,2.814837831438282e-05,1.0121735126975576e-05,2.8951505016686948e-05,2.0626871011086402e-05,3.696280205298863e-05]
    AMD_CPU_arr = [2.32208e-05,4.30732e-05,7.335889999999999e-05,0.0001188876999999,0.0001779929,0.00026363,0.0004181758999999,0.0005511653,0.0008196583999999,0.0009001752]
    AMD_CPU_std_arr = [4.675192402457892e-07,1.1130921674925817e-06,1.3209464489944735e-06,1.02022656579583e-06,1.4597906200700317e-06,1.896803053092814e-06,4.331791206238412e-06,1.96672323675702e-06,3.480220520471524e-06,4.301251506113593e-05]
    P100_arr = [0.0020051717758178,0.0018929004669189,0.0020303964614868,0.0019014120101928,0.0018664121627807,0.001814579963684,0.0018901824951171,0.001854920387268,0.0017940282821655,0.0018584489822387]
    P100_std_arr = [8.967076731630589e-05,2.5812956107875793e-05,0.0001342126231056,2.707318407289117e-05,2.547394760338819e-05,1.811058837245438e-05,3.723920464327483e-05,1.7795833117330174e-05,1.1623652208707043e-05,2.139123740028496e-05]

    # mag0_ratio_tot = np.loadtxt(f"{calc_path}/accuracy/precision_high/magn_filt_ratio_2_32.csv")

    mag0_ratio = [5.483398437500000000e-01,3.653564453125000000e-01,2.640380859375000000e-01,2.068481445312500000e-01,1.536254882812500000e-01,1.234130859375000000e-01,1.201782226562500000e-01]
    time_sample_TitanQ = []
    for i in range(len(mag0_ratio)):
        time_sample_TitanQ.append( timeout / ( 512 * mag0_ratio[i] ) )
        #bij deze hoort x_2

    plt.figure()
    plt.errorbar(x, UF_arr, yerr = UF_std_arr, capsize = 4, label = "UltraFast")
    plt.errorbar(x, AMD_CPU_arr, yerr=AMD_CPU_std_arr, capsize = 4, label="AMD CPU")
    plt.errorbar(x, P100_arr, yerr=P100_std_arr, capsize = 4, label="P100")
    plt.plot(x_2, time_sample_TitanQ, label="TitanQ")
    plt.yscale('log')
    plt.xticks(x)
    plt.xlabel("spins")
    plt.ylabel("Time per sweep (s)")
    plt.title("Time per sample for different hardware, " + r"$\alpha$" + f"=2")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.show()
makePlot_timeProjection(2)
# UF_arr =[1.4949295699999999e-05, 7.1075944e-05,0.00022066767475,0.0005290877813499999,0.0010919345298999999,0.0019848194393499996,0.0034447283587,0.00549388198105,0.00833878898185,0.0122014823203]
# print(type(UF_arr[0]))