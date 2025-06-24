import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import glob
from tqdm import tqdm
import json
import pandas as pd

from TitanQ import base_path, calc_path, param_path, bonds_path, magn_filt_ratio
from main import getVarEngVal, nspins_ls,nspins_ls_extended, alpha_ls, timeout_ls,timeout_per_n, precision_ls, load_engVal, calcRelErr_vs_timeout, trainingLoop_TQ, storeVal_path

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

# def makePlot_relErr_vs_timeout(nspins_ls, alpha, nruns):
#     figcount = 1
#     plt.figure(figcount)
#
#     #creating gradient in colours
#     def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
#         c1 = np.array(mpl.colors.to_rgb(c1))
#         c2 = np.array(mpl.colors.to_rgb(c2))
#         return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
#
#     #defining the two colours
#     c1 = 'red'  # blue
#     c2 = 'yellow'  # green
#
#     nspins_counter = 1
#     for nspins_ind in nspins_ls:
#
#         #Hier moet nog de mean bij +stddev/errorbars van de split e
#         relErr_arr = np.loadtxt(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout/relErr_{nspins_ind}_{alpha}_{nruns}.csv", delimiter = ",")
#         plt.plot(timeout_ls, relErr_arr, color=colorFader(c1, c2, nspins_counter / len(nspins_ls)), label=f'n={nspins_ind}')
#         nspins_counter += 1
#
#     # aesthetics
#     plt.xticks(timeout_ls)
#     plt.xlabel("timeout (s)")
#     plt.ylabel("Relative error")
#     plt.title(r"$\alpha$" + f"={alpha}, nruns={nruns}")
#     plt.legend()
#
#     # saving and showing the figure
#     plt.savefig(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout_a{alpha}_r{nruns}.png", bbox_inches='tight')
#     figcount += 1
#     plt.show()

def makePlot_relErr_vs_timeout_split_states(nspins_ls, alpha, timeout_ls,nruns, split_bins = 4):
    figcount = 1
    plt.figure(figcount)

    #creating gradient in colours
    def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    #defining the two colours

    c1 = '#BEF9FA'
    c2 = '#182D66'
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
    plt.xticks([0.1,2,4,10,16, 24])
    plt.ylim(-0.0008,0.026)

    plt.xlabel("timeout (s)")
    plt.ylabel("Relative error")
    plt.title(r"$\alpha$" + f"={alpha}, runs={nruns}")
    plt.legend()

    # saving and showing the figure
    plt.savefig(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout_split_{alpha}_{nruns}_lower_timeout_values.png", bbox_inches='tight')
    figcount += 1
    plt.show()


def makePlot_magn_filt_ratio(nspins_ls, alpha, timeout_ls, nruns, precision_param, split_bins = 4):
    # To get errorbar here: the function magn_filt_ratio needs to be updated so the filtered states aare stored in nruns amount of .json files for one system.
    figcount = 1
    plt.figure(figcount)

    def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    #defining the two colours
    c1 = '#182D66'
    c2 = '#BEF9FA'

    timeout_counter = 1
    for timeout in timeout_ls:
        ratio_mean_arr = []
        ratio_err_arr = []
        for nspins in nspins_ls:
        # for 4 bins:
            ratio_perTimeout_perBin_arr = []
            for split_ind in range(split_bins):

                data_filt = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/split_states/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind+1}of{split_bins}.csv", delimiter = ",")

                ratio_perTimeout_perBin_arr.append(len(data_filt) / ((nruns/split_bins) * 512))
            ratio_mean_arr.append(np.mean(ratio_perTimeout_perBin_arr))
            ratio_err_arr.append(np.std(ratio_perTimeout_perBin_arr))

        # plot for every spin value in the same plot
        plt.errorbar(nspins_ls, ratio_mean_arr, yerr = ratio_err_arr, capsize = 4, color=colorFader(c1, c2, timeout_counter / len(timeout_ls)), label = f"t={timeout}")
        timeout_counter += 1

    plt.xticks(nspins_ls)
    plt.ylim(0.025,0.725)

    plt.xlabel("nspins")
    plt.ylabel("Magnetic filter ratio")
    plt.legend()
    plt.title(r"$\alpha$" + f"={alpha}")
    plt.savefig(f"{calc_path}/accuracy/precision_high/magn_filt_ratio_{alpha}_{nruns}_errorbars.png", bbox_inches='tight')
    figcount += 1
    plt.show()


# makePlot_magn_filt_ratio(nspins_ls, 2, timeout_ls, 32, 'high')

# makePlot_magn_filt_ratio(nspins_ls, 4, timeout_ls, 32, 'high')
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
        # filt_samp_arr = varEngVal_evolution['varEngVal_evolution']

    QMC_eng = [-0.701780, -0.678872, -0.673487, 0.671549]  # 16 36 (exact) 64 100 (QMC)

    plt.figure()

    c1 = '#182D66'
    # c2 = '#BEF9FA'

    x = np.arange(epochs)

    plt.plot(x, varEngVal_arr, color = c1, label = "TQ training")
    # plt.plot(x, filt_samp_arr, label = "filt ratio")

    plt.axhline(QMC_eng[nspins_ls.index(nspins)], linestyle = 'dashed', alpha = 0.5, color = c1, label="Exact value")
    plt.ylim(-0.71,-0.51)

    myTitle = f"n={nspins}, "+ r"$\alpha$" +f"={alpha}, " + f"epochs={epochs}"#, time/sweep={int(time_per_sweep)}s"
    plt.xlabel("epoch")
    plt.ylabel("Variational Energy")
    plt.legend()

    # plt.legend(loc="upper right")
    plt.title(myTitle, loc='center', wrap=True)
    plt.savefig(f"{calc_path}/varEng/varEng_training_evolution/plots/varEngVal_Evo_{nspins}_{alpha}_{epochs}.png")

    plt.show()

# makePlot_training_varEngVal(16, 2, 300)

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

    QMC_eng = [-0.701777, -0.678873, -0.673487, 0.671549] #16 36 64 100

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    ax1.plot(x, varEngVal_arr, color=varEng_col, lw=3)
    ax1.axhline(QMC_eng[nspins_ls.index(nspins)],label = "QMC Eng")
    ax2.plot(x, filt_samp_arr, color=filt_col, lw=4)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("variational Energy", color=varEng_col, fontsize=14)
    ax1.tick_params(axis="y", labelcolor=varEng_col)

    ax2.set_ylabel("Amount of filtered samples", color=filt_col, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=filt_col)

    fig.suptitle("VarEng & filt samples vs epochs", fontsize=20)
    fig.autofmt_xdate()

    # plt.savefig(f"{calc_path}/varEng/varEng_training_evolution/plots/varEngVal_Evo_and_filtSamps_{nspins}_{alpha}_{epochs}.png")

    plt.show()

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
    plt.errorbar(x, runtime_new_avg, yerr=runtime_new_std, capsize = 4, color = c2, linestyle='dotted', label="New, " + r"$\alpha$" + f"={2}")
    plt.errorbar(x, runtime_avg_4, yerr=runtime_std_4, alpha = 0.8, capsize=4, color=c1, lw = 0.8, label="Old, " + r"$\alpha$" + f"={4}")
    plt.errorbar(x, runtime_new_avg_4, yerr=runtime_new_std_4, alpha = 0.8, capsize=4, color=c2, lw = 0.8, label="New, " + r"$\alpha$" + f"={4}")
    plt.yscale('log')
    plt.xticks(x)
    plt.xlabel("nspins")
    plt.ylabel("Computation time per filtered sample (s)")
    plt.title("Computational advantage of using lookup table")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.savefig(f"{calc_path}/varEng/locEng_calc_methods_comp.png")
    plt.show()

# makePlot_locEng_speedup(2, 32, 'high')

def makePlot_sketch_timeout():

    x=np.arange(0,10,0.01)
    y=np.exp(-(x-2))+0.5
    y_2 = 0.5

    c1 = '#182D66'

    plt.figure()
    plt.plot(x, y, color = c1, label="TitanQ")
    plt.axhline(0.5, color = c1, linestyle = 'dashed', alpha = 0.5, label ="balancing line")
    plt.xlabel("Time or samples")
    plt.ylabel("Energy of state")
    plt.title("Sketch of thermalisation of TitanQ")
    plt.legend()
    plt.gca().axes.get_yaxis().set_ticklabels([])
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.savefig(f"{calc_path}/Thermalisation_sketch.png")
    plt.show()

def makePlot_timeProjection(alpha = 2, timeout = 2, precision_param = 'high', nruns = 32, split_bins = 4):
    # To get errorbar here: the function magn_filt_ratio needs to be updated so the filtered states aare stored in nruns amount of .json files for one system.

    # data = np.loadtxt(f"{base_path}/projections-MH-Ising-FPGA-Conservative-Optimistic.csv", delimiter = ",", skiprows=8)
    x = [16,36,64,100,144,196,256,324,400,484]
    x_2 = nspins_ls

    # df = pd.read_csv(f"{base_path}/projections-MH-Ising-FPGA-Conservative-Optimistic.csv")
    UF_arr = [1.4949295699999999e-05,7.1075944e-05,0.00022066767475,0.0005290877813499999,0.0010919345298999999,0.0019848194393499996,0.0034447283587,0.00549388198105,0.00833878898185,0.0122014823203]
    UF_sem_arr = [1.281692194360984e-07,2.400770237115583e-07,6.494463287045599e-07,1.4500511100564922e-06,4.320382517891083e-06,2.814837831438282e-05,1.0121735126975576e-05,2.8951505016686948e-05,2.0626871011086402e-05,3.696280205298863e-05]
    c2 = '#4D658B'

    AMD_arr = [6.024429382163056e-05,8.507603140821202e-05,0.00021346740704936235,0.0005386906842602199,0.0010351553123387613,0.0018829416907445874,0.004781920065398177,0.004304961784868221,0.010247831520274859,0.011624627944487226]
    AMD_sem_arr = [6.664705163286599e-06,9.723958959340587e-06,3.011601469151001e-05,9.938254550254025e-05,6.46945940711511e-05,0.00029142842094319706,0.0007517743788652227,0.0003010697388033933,0.0019289404208148555,0.001732008941557843]
    c3 = '#6783A2'

    ASIC_arr = [1.0377643116797107e-05,7.900600039765983e-06,1.1639618753790604e-05,1.8124353798102683e-05,2.3262845031206557e-05,2.856946008792e-05,4.5740752304466335e-05,3.124261839319871e-05,5.001025559074907e-05,5.16549575881994e-05]
    ASIC_sem_arr = [1.1288848071356669e-06,8.796340092548321e-07,1.6286888603759943e-06,3.3401260170205997e-06,1.4412964089971667e-06,4.416998899675706e-06,7.1753600712043175e-06,2.182122810303056e-06,9.41099172758859e-06,7.2898103426467305e-06]
    c4 = '#80A1B8'

    FPGA_arr = [3.706301113141823e-07,2.821642871344994e-07,4.157006697782358e-07,6.472983499322386e-07]#,8.308158939716626e-07,1.020337860282857e-06,1.6335982965880832e-06,1.1158077997570969e-06,1.7860805568124667e-06,1.8448199138642642e-06]
    FPGA_sem_arr = [4.031731454055953e-08,3.1415500330529714e-08,5.816745929914265e-08,1.1929021489359283e-07]#,5.147487174989881e-08,1.5774996070270378e-07,2.5626285968586845e-07,7.793295751082342e-08,3.361068474138782e-07,2.603503693802404e-07]
    c5 = '#99BECE'

    # Fast_arr = [1.0377643116797106e-07,7.900600039765983e-08,1.1639618753790603e-07,1.812435379810268e-07,2.3262845031206555e-07,2.856946008792e-07,4.5740752304466334e-07,3.1242618393198715e-07,5.001025559074907e-07,5.16549575881994e-07]
    # Fast_sem_arr = [1.1288848071356668e-08,8.796340092548321e-09,1.6286888603759943e-08,3.3401260170205996e-08,1.4412964089971667e-08,4.416998899675706e-08,7.175360071204317e-08,2.1821228103030557e-08,9.41099172758859e-08,7.289810342646731e-08]
    c6 = '#B0D9E2'


    # mag0_ratio = [5.483398437500000000e-01,3.653564453125000000e-01,2.640380859375000000e-01,2.068481445312500000e-01,1.536254882812500000e-01,1.234130859375000000e-01,1.201782226562500000e-01]
    time_sample_TitanQ_mean_arr = []
    time_sample_TitanQ_err_arr = []
    # for every spin values
    for nspins in nspins_ls:

        # for 4 bins:
        ratio_perSpin_perBin_arr = []
        for split_ind in range(split_bins):

            data_filt = np.loadtxt(f"{calc_path}/filt_states/precision_{precision_param}/split_states/vis_states_filt_{nspins}_{alpha}_{timeout}_{nruns}_{split_ind+1}of{split_bins}.csv", delimiter = ",")

            ratio_perSpin_perBin_arr.append(len(data_filt) / ((nruns/split_bins) * 512))


        time_sample_TitanQ_perSpin = []
        for i in range(len(ratio_perSpin_perBin_arr)): #was mag0_ratio ipv ratio_mean_arr
            time_sample_TitanQ_perSpin.append( timeout / ( 512 * ratio_perSpin_perBin_arr[i] ) )
        #bij deze hoort x_2

        # dan pakken we van die vier waarden de gemiddelde en de fout voor elke spin waarde.
        time_sample_TitanQ_mean_arr.append(np.mean(time_sample_TitanQ_perSpin))
        time_sample_TitanQ_err_arr.append(np.std(time_sample_TitanQ_perSpin))
    plt.figure()
    plt.errorbar(x, UF_arr, yerr = UF_sem_arr, capsize = 4, color = c2, linestyle = 'dashed',  label = "UltraFast")
    plt.errorbar(x, AMD_arr, yerr=AMD_sem_arr, capsize = 4, color = c3, label="AMD CPU", )
    plt.errorbar(x, ASIC_arr, yerr=ASIC_sem_arr, capsize = 4, color = c4, linestyle = 'dashdot', label="ASIC")
    plt.errorbar([16,36,64,100], FPGA_arr, yerr = FPGA_sem_arr, capsize = 4, color=c5, linestyle = 'dashed',label="FPGA")
    # plt.errorbar(x, Fast_arr, yerr=Fast_sem_arr, capsize = 4, color = c6,label="Fast")
    plt.errorbar(x_2, time_sample_TitanQ_mean_arr, yerr = time_sample_TitanQ_err_arr, capsize = 4, color = '#182D66', label="TitanQ")

    plt.yscale('log')
    plt.xticks(x)
    plt.ylim(4e-8,6e-1)

    plt.xlabel("spins")
    plt.ylabel("Time per sweep (s)")
    plt.title("Time per sample for different methods, " + r"$\alpha$" + f"=2")


    plt.legend(loc = 'upper left', ncol = 4)
    plt.grid(alpha = 0.5)
    plt.savefig(f"{calc_path}/varEng/time_projection_with_errorbars.png")
    plt.show()

makePlot_timeProjection(alpha = 2, timeout = 2, precision_param = 'high', nruns = 32, split_bins = 4)

def makePlot_sampsTaken_vs_timeout(nspins_ls, alpha, timeout_ls, precision_param, nruns = 32):


    def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    #defining the two colours
    c2 = '#182D66'
    c1 = '#BEF9FA'

    figcount = 1
    plt.figure(figcount)
    nspins_counter = 1
    for nspins in tqdm(nspins_ls):
        samps_taken_mean_arr = []
        samps_taken_err_arr = []
        for timeout_ind in timeout_ls:
            samps_taken_timeout_arr = []
            for nruns_ind in range(nruns):

                with open(f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout_ind}/TQ_states_{nspins}_{alpha}_{timeout_ind}_{nruns_ind+1}.json", 'r') as file:
                    data = json.load(file)
                    samps_taken = data['samps_taken']
                # samps_taken_mean += samps_taken
                samps_taken_timeout_arr.append(samps_taken)
            samps_taken_mean_arr.append(np.mean(samps_taken_timeout_arr))
            samps_taken_err_arr.append(np.std(samps_taken_timeout_arr))
            # samps_taken_mean_arr.append(samps_taken_mean/nruns)


        plt.errorbar(timeout_ls, samps_taken_mean_arr, yerr = samps_taken_err_arr, capsize = 4, color = colorFader(c1, c2, nspins_counter / len(nspins_ls)), label = f"n={nspins}")
        nspins_counter += 1
    plt.xlabel("timeout (s)")
    plt.xticks([0.1,2,4,10,16,24])
    plt.ylabel("Samples taken")
    plt.title(r"$\alpha$" + f"={alpha}")

    plt.legend()

    plt.savefig(f"{calc_path}/sampsTaken_vs_timeout_a{alpha}.png")
    figcount += 1

    plt.show()

def makePlot_spins_vs_timePerSample(alpha_ls, precision_param, nruns = 32):
    c1 = '#182D66'
    c6 = '#B0D9E2'
    col = [c1,c6]
    figcount = 1
    plt.figure(figcount)
    nspins_ls_ls = [nspins_ls_extended, nspins_ls]

    for alpha in alpha_ls:
        timePerSample_arr = []
        err = []
        for nspins in nspins_ls_ls[alpha_ls.index(alpha)]:
            # timePerSample_sum = 0
            timePerSample_nspin = []
            for nruns_ind in range(nruns):
                with open(f"{calc_path}/states/precision_{precision_param}/all_states_{nspins}_{alpha}_{timeout_per_n[nspins_ls_ls[alpha_ls.index(alpha)].index(nspins)]}/TQ_states_{nspins}_{alpha}_{timeout_per_n[nspins_ls_ls[alpha_ls.index(alpha)].index(nspins)]}_{nruns_ind + 1}.json",'r') as file:
                    data = json.load(file)
                    samps_taken = data['samps_taken']
                # timePerSample_sum += timeout/samps_taken
                timePerSample_nspin.append(timeout_per_n[nspins_ls_ls[alpha_ls.index(alpha)].index(nspins)]/samps_taken)
            timePerSample_mean = np.mean(timePerSample_nspin)
            timePerSample_arr.append(timePerSample_mean)
            err.append(np.std(timePerSample_nspin))
        plt.errorbar(nspins_ls_ls[alpha_ls.index(alpha)], timePerSample_arr, yerr = err, capsize = 4, color = col[alpha_ls.index(alpha)], label = r"$\alpha$"+f"={alpha}")

    plt.xlabel("nspins")
    # plt.xticks(nspins_ls_extended)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-3,3e-2)
    plt.ylabel("time per sample (s)")
    # plt.title(f"timeout={timeout}s")
    plt.tight_layout()

    plt.legend(loc = 'upper left')

    plt.savefig(f"{calc_path}/nspins_vs_timePerSample_extended.png")
    figcount += 1

    plt.show()
