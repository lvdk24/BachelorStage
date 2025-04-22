Filenames/naamgeving:
In states you can find the folder:
all_states_{nspins}_{alpha}_{timeout}
	TQ_states_{nspins}_{alpha}_{timeout}_{nruns_ind+1}.json

Where nruns_ind+1 indicates the zoveelste nrun (one nrun is defined by sampling 512 states)

RBMEng
	RBMEng_{nspins}_{alpha}_{timeout}.csv
	RBMEngPlots
		RBMEngPlot_comp_{nspins}_{alpha}_{timeout}.png
		or:
		RBMEngPlot_comp_{nspins}_{alpha}_{timeout}_{nruns}.png
		(By default nruns will be equal to 8)
Accuracy
	relErr_vs_timeout_{alpha}_{nruns}
		spins are accounted for in the plot. RelErr value per timeout value is noted in 		the .csv file.
	
Voor varEng en locEng folders geldt ook:
varEng_{nspins}_{alpha}_{timeout}_{nruns}
locEng_{nspins}_{alpha}_{timeout}_{nruns}


Plotting:
With making plots and reading the energy values from UltraFast, I used what was called the local Energy as the variational energy. Naming convention error here.

The states calculated by TitanQ are standard with high precision, unless stated otherwise. This can be found in the .json file data as well.

