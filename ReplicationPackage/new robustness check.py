#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_stata("ms_blel_jpal_long.dta")

# Baseline rows
base = df[df["round"] == "Baseline"][[
    "st_id", "strata", "treat", "m_theta_mle", "h_theta_mle", "in_r2"
]].rename(columns={
    "m_theta_mle": "m_base",
    "h_theta_mle": "h_base",
    "in_r2": "in_r2_base"
})

# Endline rows
end = df[df["round"] == "Endline"][[
    "st_id", "m_theta_mle", "h_theta_mle", "in_r2"
]].rename(columns={
    "m_theta_mle": "m_end",
    "h_theta_mle": "h_end",
    "in_r2": "in_r2_end"
})

# Merge into wide format
wide = base.merge(end, on="st_id")

# Restrict to IRT linking sample
wide = wide[(wide["in_r2_base"] == 1) & (wide["in_r2_end"] == 1)]


# In[2]:


# compute z-scores using control baseline mean & SD
control_base = wide[wide["treat"] == 0]

for subj in ["m_base", "m_end", "h_base", "h_end"]:
    mu = control_base[subj].mean()
    sd = control_base[subj].std()
    wide[subj + "_z"] = (wide[subj] - mu) / sd


# In[3]:


def itt_reg(y_col, base_col):
    data = wide[[y_col, base_col, "treat", "strata"]].dropna().copy()

    X = data[["treat", base_col]]

    # strata fixed effects
    strata_dummies = pd.get_dummies(
        data["strata"].astype(int), prefix="strata", drop_first=True
    )
    X = pd.concat([X, strata_dummies], axis=1)

    X = sm.add_constant(X)
    y = data[y_col]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model


# In[4]:


mod_math_z = itt_reg("m_end_z", "m_base_z")
mod_hindi_z = itt_reg("h_end_z", "h_base_z")

print("Math ITT (Z-score):", mod_math_z.params["treat"])
print("Hindi ITT (Z-score):", mod_hindi_z.params["treat"])

print("\nFull results – Math Z:\n", mod_math_z.summary().tables[1])
print("\nFull results – Hindi Z:\n", mod_hindi_z.summary().tables[1])


# In[ ]:




