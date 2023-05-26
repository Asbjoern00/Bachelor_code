import matplotlib.pyplot as plt
import pandas as pd
import experiment_utils as utils
import gc
path = 'C:/Users/andre/Bachelor_code/experiment_results/gw_ucrl_l'

for ns in [100]:
    df = pd.read_pickle(f"/home/asr/Desktop/Bachelor/Code bachelor/numba_code/experiment_results/gw_ucrl_l/nS_{ns}_correct_confidence.pkl")
    ucrl_smdp_l = df.loc[:, df.columns.str.startswith('UCRL_SMDP-L')]
    ucrl_smdp = df.loc[:, (df.columns.str.startswith('UCRL_SMDP')) & ~(df.columns.str.startswith('UCRL_SMDP-L'))]
    ucrl_l = df["UCRL2-L, T_max = 1"]
    ratio = ucrl_smdp_l.iloc[10**7-1]/ucrl_l.iloc[10**7-1]
    ratio = ratio.reset_index()
    ratio =ratio.rename(columns={9999999:"Ratio of regret"})
    ratio["T_max"] = ratio["index"].str.split('= ').str[-1].astype(int)

    del df
    gc.collect()

    utils.plot_mean_regret_from_pd(ucrl_smdp_l, nS=ns, save_to=f"ucrl_smdp_l_ns_{ns}.png")
    del ucrl_smdp_l
    gc.collect()
    utils.plot_mean_regret_from_pd(ucrl_smdp, nS=ns, save_to=f"ucrl_smdp_ns_{ns}.png")
    del ucrl_smdp
    gc.collect()

    ax = ratio.plot(x = "T_max", y = "Ratio of regret", kind = "line")
    ax.grid()
    fig = ax.get_figure()
    fig.savefig(f"ratio_of_regrets{ns}.png")
    plt.clf()
    plt.close()