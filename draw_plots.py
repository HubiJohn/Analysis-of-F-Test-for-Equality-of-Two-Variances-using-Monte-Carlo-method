### This file contains an example of two methods of creating 

import numpy as np
from scipy.stats import f
from scipy.stats import norm
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from get_errors import *
from ftest_setup import *


#Testing out different number of simulations
different_m = [10, 100, 1000, 10_000, 100_000]

testing_m_df = pd.DataFrame({"m" : [],
                          "p_hat" : [],
                          "se_hat" : [],
                          "duration[s]" : []})

for m in different_m:
    start = time.time()
    p_hat, se_hat = get_type_I_error(m=m)
    end = time.time()
    testing_m_df.loc[len(testing_m_df)] = {"m" : m,
                                          "p_hat" : p_hat,
                                          "se_hat" : se_hat,
                                          "duration[s]" : end-start}
    


#PLOT
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#type I empirical probability
sns.barplot(data=testing_m_df,
            x="m",
            y="p_hat",
            palette="Purples",
            ax=axes[0])
axes[0].set_title("Empirical Type I Error Probability")
axes[0].set_ylabel('')
axes[0].set_xlabel("Number of Simulations (m)")

x_coords = [p.get_x() + 0.5 * p.get_width() for p in axes[0].patches]
y_coords = [p.get_height() for p in axes[0].patches]
axes[0].errorbar(x=x_coords, y=y_coords, yerr=testing_m_df["se_hat"], fmt="none", c="k")

#duration
sns.barplot(data=testing_m_df,
            x="m",
            y="duration[s]",
            palette="Greens",
            ax=axes[1])
axes[1].set_title("Calculations Duration [s]")
axes[1].set_ylabel('')
axes[1].set_xlabel("Number of Simulations (m)")

plt.tight_layout()
plt.show()



#Testing out different sample sizes

different_n = [5, 10, 30, 50, 100, 500, 1000]

testing_n_df = pd.DataFrame({"n" : [],
                          "p_hat" : [],
                          "se_hat" : []})

for n in different_n:
    p_hat, se_hat = get_type_I_error(m=10_000, n=n)
    testing_n_df.loc[len(testing_n_df)] = {"n" : n,
                                          "p_hat" : p_hat,
                                          "se_hat" : se_hat}
    
#PLOT
ax = sns.barplot(data=testing_n_df,
                 x="n",
                 y="p_hat",
                 palette="Blues")

ax.set_ylabel("Empirical Type I Error Probability")
ax.set_xlabel("Sample size (n)")

x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_coords, y=y_coords, yerr=testing_n_df["se_hat"], fmt="none", c="k")

plt.show()