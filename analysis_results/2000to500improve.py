import pandas as pd

# 更新后的表格数据
data = {
    ("P3", 500): {"Funsearch": 7042.20, "EoH": 7268.21, "ReEvo": 6564.44, "ParEvo": 6539.23},
    ("P3", 2000): {"Funsearch": 6656.70, "EoH": 6749.50, "ReEvo": 6506.51, "ParEvo": 6479.49},
    ("P4", 500): {"Funsearch": 24289.3, "EoH": 19152.4, "ReEvo": 11102.32, "ParEvo": 6424.6},
    ("P4", 2000): {"Funsearch": 15646.1, "EoH": 14235.5, "ReEvo": 6038.45, "ParEvo": 5489.0},
}

# 转换成 DataFrame
df = pd.DataFrame(data).T
df.index.names = ["Benchmark", "Budget"]

print("原始数据：")
print(df)

# 计算提升率
results = {}
methods = df.columns

for method in methods:
    p3_500, p3_2000 = df.loc[("P3", 500), method], df.loc[("P3", 2000), method]
    p4_500, p4_2000 = df.loc[("P4", 500), method], df.loc[("P4", 2000), method]

    imp_p3 = (p3_500 - p3_2000) / p3_500 * 100
    imp_p4 = (p4_500 - p4_2000) / p4_500 * 100
    avg_imp = (imp_p3 + imp_p4) / 2

    results[method] = {"P3 Improvement (%)": imp_p3,
                       "P4 Improvement (%)": imp_p4,
                       "Average Improvement (%)": avg_imp}

# 转成 DataFrame
improvement_df = pd.DataFrame.from_dict(results, orient="index")
print("\n提升率结果：")
print(improvement_df.round(2))
