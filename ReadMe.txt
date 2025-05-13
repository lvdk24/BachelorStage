Filenames/naamgeving:

25/4/25 16h00:
nspins_ls = [16,36,64,100,196,324,484]
alpha_ls = [2,4]
timeout_ls = [2,4,10,16, 24]#, 32]
precision_ls = ['standard','high']


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
		spins are accounted for in the plot. RelErr value per timeout value is noted in the .csv file.
	
Voor varEng en locEng folders geldt ook:
varEng_{nspins}_{alpha}_{timeout}_{nruns}
locEng_{nspins}_{alpha}_{timeout}_{nruns}


Plotting:
With making plots and reading the energy values from UltraFast, I used what was called the local Energy as the variational energy. Naming convention error here.

The states calculated by TitanQ are standard with high precision, unless stated otherwise. This can be found in the .json file data as well.


For running code on the cluster: (cn38)
1. Open two terminal windows, local and cluster
2. Go to directory of python file in local and cluster window
	2a. For local:
		pwd #(to see which directory you're in)
		cd PycharmProjects/RadboudProgrammeren/BachelorStage/
	2b. For cluster: 
		ssh lilo
		Voer wachtwoord in
		ssh cn38
		pwd #(to see which directory you're in)
		cd Stage/BachelorStage
3. For getting local code to the cluster:
	3a. Commit to main
	3b. Push to production #(or origin, but I renamed it to production)
	3c. Check if the code is up to date in local terminal by "git push"
	3d. In cluster terminal, "git pull"
4a. To sync files from local to cluster: "source sync_to_cn38.sh"
4b. To sync files from cluster to local: "source sync_38.sh" (both these commands merge, no overwrite)
5. To run the code on the cluster with properly installed pip modules:
	5a. In cluster terminal, in the project directory: "source /scratch/envs/activate_micromamba.sh"
	5b. "micromamba activate"
	5c. To run the desired code: "python3 your_code.py", in my case: "python3 main.py"
	

