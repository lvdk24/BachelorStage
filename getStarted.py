import os

#fill in your own base_path (in the .env file)

base_path = os.getenv("BASE_PATH")

calc_path = f"{base_path}/calculations"
param_path = f"{base_path}/ising_params"
bonds_path = f"{base_path}/bonds"
storeVal_path = f"{calc_path}/varEng/varEng_training_evolution"

def makeDirectories():

    os.makedirs(bonds_path)

    os.makedirs(calc_path)

    os.makedirs(param_path)

    os.makedirs(storeVal_path)

    os.makedirs(f"{calc_path}/varEng/precision_high")
    os.makedirs(f"{calc_path}/varEng/precision_standard")

    os.makedirs(f"{calc_path}/states/precision_high")
    os.makedirs(f"{calc_path}/states/precision_standard")

    os.makedirs(f"{calc_path}/RBMEng/precision_high")
    os.makedirs(f"{calc_path}/RBMEng/precision_standard")

    os.makedirs(f"{calc_path}/locEng/precision_high")
    os.makedirs(f"{calc_path}/locEng/precision_standard")
    os.makedirs(f"{calc_path}/locEng/precision_high/split_states")
    os.makedirs(f"{calc_path}/locEng/precision_standard/split_states")

    os.makedirs(f"{calc_path}/filt_states/precision_high")
    os.makedirs(f"{calc_path}/filt_states/precision_standard")
    os.makedirs(f"{calc_path}/filt_states/precision_high/split_states")
    os.makedirs(f"{calc_path}/filt_states/precision_standard/split_states")

    os.makedirs(f"{calc_path}/accuracy/precision_high")
    os.makedirs(f"{calc_path}/accuracy/precision_standard")
    os.makedirs(f"{calc_path}/accuracy/precision_high/relErr_vs_nspins")
    os.makedirs(f"{calc_path}/accuracy/precision_high/relErr_vs_timeout")


