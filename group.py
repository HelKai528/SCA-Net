import pandas as pd
import numpy as np
import sys
import os


def process_csv(input_file,train):
    df = pd.read_csv(input_file, sep=',')

    # 根据Teff分为四组
    a_type = df[df['teff'] > 7500]  # a型：7500以上
    f_type = df[(df['teff'] >= 6000) & (df['teff'] <= 7500)]  # f型：6000-7500
    g_type = df[(df['teff'] >= 5000) & (df['teff'] < 6000)]  # g型：5000-6000
    k_type = df[df['teff'] < 5000]  # k型：5000以下

    print(f"各组样本数量: a型({len(a_type)}), f型({len(f_type)}), g型({len(g_type)}), k型({len(k_type)})")
    # min_count = min(len(a_type), len(f_type), len(g_type), len(k_type))
    min_count=100000000000
    # 随机抽样（如果样本不足则全部保留）
    a_sampled = a_type.sample(n=min_count, replace=False) if len(a_type) > min_count else a_type
    f_sampled = f_type.sample(n=min_count, replace=False) if len(f_type) > min_count else f_type
    g_sampled = g_type.sample(n=min_count, replace=False) if len(g_type) > min_count else g_type
    k_sampled = k_type.sample(n=min_count, replace=False) if len(k_type) > min_count else k_type

    if train == True:
        output_dir = "/home81/haokai/lamost/groupcsv/train"
    else:
        output_dir = "/home81/haokai/lamost/groupcsv/test"
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    a_sampled.to_csv(f"{output_dir}/a_type.csv", index=False, sep=',')
    f_sampled.to_csv(f"{output_dir}/f_type.csv", index=False, sep=',')
    g_sampled.to_csv(f"{output_dir}/g_type.csv", index=False, sep=',')
    k_sampled.to_csv(f"{output_dir}/k_type.csv", index=False, sep=',')

    print(f"处理完成！样本分组数量：")
    print(f"  a型: {len(a_sampled)} 个样本")
    print(f"  f型: {len(f_sampled)} 个样本")
    print(f"  g型: {len(g_sampled)} 个样本")
    print(f"  k型: {len(k_sampled)} 个样本")
    print(f"结果已保存到 {output_dir}/ 目录")


