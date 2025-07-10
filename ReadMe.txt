To get started: run the getStarted.py file. This will create all the necessary directories used in the code.

Naming: Since we have used quite a lot of parameters and want to distinguish between those in the filename for easy reading, we use the following naming system:
n - nspins, a - alpha, t - timeout, r - nruns (or ri for nruns index). For example:
TQ_states_n16_a2_t0.1_ri1.json.

Have fun :)


----------------
Extra information
----------------

For running code on the cluster: (cn38)
1. Open two terminal windows, local and cluster
2. Go to directory of python file in local and cluster window
	2a. For local:
		"pwd" #(to see which directory you're in)
		"cd [YOURDIRECTORY]"
	2b. For cluster: 
		ssh lilo
		Enter Password
		ssh cn38
		"pwd" #(to see which directory you're in)
		"cd [YOURDIRECTORY]"
3. For getting local code to the cluster:
	3a. Commit to main
	3b. Push to main (or whatever branch you named)
	3c. Check if the code is up to date in local terminal by "git push"
	3d. In cluster terminal, "git pull"
4a. To sync files from local to cluster: "source sync_to_cn38.sh"
4b. To sync files from cluster to local: "source sync_38.sh" (both these commands merge, there is no deleting of files)
5. To run the code on the cluster with properly installed pip modules:
	5a. In cluster terminal, in the project directory: 
		"source /scratch/envs/activate_micromamba.sh"
	5b. "micromamba activate"
	5c. To run the desired code: "python3 your_code.py" (perhaps without the "3" for 			windows/linux)
Run tmux to still be able to access the computercluster:
	To get out of tmux: ^b d
	To get back into tmux: tmux a