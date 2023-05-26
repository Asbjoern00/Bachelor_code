import matplotlib.pyplot as plt
import pandas as pd
import experiment_utils as utils
import gc
path = '/home/asr/Desktop/Bachelor/Code bachelor/numba_code/experiment_results/rs_ucrl_l'

fig, ax = plt.subplots()

for ns in [10,15,20,25,50]:

    df = pd.read_pickle(f"/home/asr/Desktop/Bachelor/Code bachelor/numba_code/experiment_results/rs_ucrl_l/nS_{ns}_correct_confidence.pkl")
    ucrl_smdp_l = df.loc[:, (df.columns.str.startswith('UCRL_SMDP-L')) | (df.columns.str.startswith('UCRL2-L'))]
    ucrl_smdp = df.loc[:, (df.columns.str.startswith('UCRL_SMDP')) & ~(df.columns.str.startswith('UCRL_SMDP-L'))]
    ucrl_l = df["UCRL2-L, T_max = 1"]
    ratio = ucrl_smdp_l.iloc[10**7-1]/ucrl_l.iloc[10**7-1]
    ratio = ratio.reset_index()
    ratio =ratio.rename(columns={9999999:"Ratio of regret"})
    ratio["T_max"] = ratio["index"].str.split('= ').str[-1].astype(int)
    ratio["index"] = ns
    ratio = ratio.sort_values("T_max")
    ratio.plot(x = "T_max", y = "Ratio of regret", label = f"{ns} states", kind = "line", ax = ax)

ax.legend()
ax.grid()
ax.set_ylabel("Ratio of regret")
fig.savefig("Ratio_of_regrets.png")
plt.clf()
plt.close()

for ns in [10,15,20,25,50]:

    df = pd.read_pickle(f"/home/asr/Desktop/Bachelor/Code bachelor/numba_code/experiment_results/rs_ucrl_l/nS_{ns}_correct_confidence.pkl")
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