import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from TitanQ import base_path, calc_path, param_path, bonds_path
from main import getVarEngVal, plotting, nspins_ls, alpha_ls, timeout_ls, precision_ls, load_engVal

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
    myTitle = f"Difference in RBM Energy TQ vs UF, n={nspins}, alpha={alpha}, timeout = {timeout}s, precision = {precision_param}"
    plt.title(myTitle, loc='center', wrap=True)
    plt.legend(loc="upper right")

    plt.savefig(f"{calc_path}/RBMEng/precision_{precision_param}/RBMEngPlots/RBMEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

def make_RBMEng_diff_prec_plot(nspins, alpha, timeout, nruns, precision_ls):
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
    myTitle = f"Difference in RBM Energy TQ vs UF, n={nspins}, alpha={alpha}, timeout = {timeout}s"
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
    myTitle = f"Difference in variational Energy TQ vs UF, n={nspins}, alpha={alpha}, timeout = {timeout}s, nruns = {nruns}, precision = {precision_param}"
    plt.xlabel("Variational Energy")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")
    plt.title(myTitle, loc='center', wrap=True)
    plt.savefig(f"{calc_path}/varEng/precision_{precision_param}/varEngPlots/varEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png")
    figcount += 1
    plt.show()

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
    myTitle = f"Difference in variational Energy TQ vs UF, n={nspins}, alpha={alpha}, timeout = {timeout}s, nruns = {nruns}"
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
        plt.plot(nspins_ls, relErr_arr, color = col[col_ind], label=f'alpha = {alpha}')
        col_ind += 1

    #aesthetics
    plt.xticks(nspins_ls)
    plt.xlabel("nspins")
    plt.ylabel("Relative error")
    plt.title("Relative error, nruns = 8, timeout = 10")
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

    nspins_ind = 1
    for nspins in nspins_ls:

        relErr_arr = np.loadtxt(f"{calc_path}/accuracy/relErr_vs_timeout/relErr_{nspins}_{alpha}_{nruns}.csv", delimiter = ",")
        plt.plot(timeout_ls, relErr_arr, color=colorFader(c1, c2, nspins_ind / len(nspins_ls)), label=f'nspins = {nspins}')
        nspins_ind += 1

    # aesthetics
    plt.xticks(timeout_ls)
    plt.xlabel("timeout (s)")
    plt.ylabel("Relative error")
    plt.title(f"Relative error for alpha={alpha} and nruns = 8")
    plt.legend()

    # saving and showing the figure
    plt.savefig(f"{calc_path}/accuracy/relErr_vs_timeout_{alpha}_{nruns}.png", bbox_inches='tight')
    figcount += 1
    plt.show()



# making distributions differences plots
if plotting:
    for alpha in alpha_ls:
        for nspins in nspins_ls:
        # make_relErr_vs_timeout_plot(nspins_ls, alpha, 8)
        #     make_RBMEng_diff_plot(nspins, alpha, 10, 8, 'standard')
        #     make_varEng_diff_plot(nspins, alpha, 10, 8, 'standard')
            make_varEng_diff_prec_plot(nspins, alpha, 10, 8, precision_ls)
            # make_RBMEng_diff_plot(nspins,alpha,10,8)
    # make_RBMEng_diff_plot(64, 2, 60, 8)
    # make_RBMEng_diff_plot(64, 2, 10, 32)
    # make_varEng_diff_plot(36, 2, 10, 32)
else:
    print("Plotting is turned off")

# make_RBMEng_diff_plot(16,2,10,8,'high')
