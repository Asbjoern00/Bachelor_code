import matplotlib.pyplot as plt
import pandas as pd
import experiment_utils as utils
import gc
#path = 'C:/Users/andre/Bachelor_code/experiment_results/gw_ucrl_l'

<<<<<<< HEAD

for ns in [10*10,15*15]:

    df = pd.read_pickle(f"/experiment_results/gw_ucrl_l/nS_{ns}_correct_confidence.pkl")
=======
for ns in [100,225]:

    df = pd.read_pickle("C:/Users/andre/Bachelor_code/experiment_results/gw_ucrl_l/nS_{ns}_correct_confidence.pkl")
>>>>>>> 326dd3b8307724b72ddaad6568260b7d5c9deadf
    ucrl_smdp_l = df.loc[:, (df.columns.str.startswith('UCRL_SMDP-L')) | (df.columns.str.startswith('UCRL2-L'))]
    ucrl_smdp = df.loc[:, (df.columns.str.startswith('UCRL_SMDP')) & ~(df.columns.str.startswith('UCRL_SMDP-L')) |  ((df.columns.str.startswith('UCRL2'))&~(df.columns.str.startswith('UCRL2-L')))]

    del df
    gc.collect()

    utils.plot_mean_regret_from_pd(ucrl_smdp_l, nS=ns, save_to=f"ucrl_smdp_l_ns_{ns}.png")
    del ucrl_smdp_l
    gc.collect()
    utils.plot_mean_regret_from_pd(ucrl_smdp, nS=ns, save_to=f"ucrl_smdp_ns_{ns}.png")
    del ucrl_smdp
    gc.collect()