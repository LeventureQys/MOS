from DNSMOS import dnsmos_local  # 16_000
from NISQA_master.nisqa.NISQA_model import nisqaModel
from mos_finetune_ssl_main import predict_noGT  # 16_000
from sigmos.sigmos import SigMOS  # 48_000
import librosa
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def process_waveform(waveform, sampling_rate,wavpath):
    # DNSMOS
    args_dnsmos = argparse.Namespace(
        testset_dir=wavpath,
        csv_path="./DNSMOS.csv",
        personalized_MOS=True,
        sampling_rate=sampling_rate,
    )
    df_dnsmos = dnsmos_local.main(args_dnsmos).drop(
        ["filename", "sr", "num_hops"], axis=1
    )

    # NISQA
    args_nisqa = argparse.Namespace(
        mode="predict_file",
        pretrained_model="./NISQA_master/weights/nisqa.tar",
        deg=waveform,
        ms_channel=1,
        output_dir=None,
    )
    df_nisqa = nisqaModel(vars(args_nisqa)).predict().drop(["deg"], axis=1)

    # mosssl
    args_mosssl = argparse.Namespace(
        fairseq_base_model="./mos_finetune_ssl_main/fairseq/wav2vec_small.pt",
        datadir=waveform,
        finetuned_checkpoint="./mos_finetune_ssl_main/pretrained/ckpt_w2vsmall",
        outfile="./mos_finetune_ssl_main/answer.txt",
    )
    df_mosssl = predict_noGT.main(args_mosssl)

    # sigmos
    # sigmos_estimator = SigMOS(model_dir="sigmos")
    # df_sigmos = pd.DataFrame(
    #     columns=[
    #         "MOS_COL",
    #         "MOS_DISC",
    #         "MOS_LOUD",
    #         "MOS_NOISE",
    #         "MOS_REVERB",
    #         "MOS_SIG",
    #         "MOS_OVRL",
    #     ]
    # )
    # sigmos_results = sigmos_estimator.run(waveform, sr=sampling_rate)
    # df_sigmos = df_sigmos._append(
    #     pd.Series(
    #         {
    #             "MOS_COL": sigmos_results["MOS_COL"],
    #             "MOS_DISC": sigmos_results["MOS_DISC"],
    #             "MOS_LOUD": sigmos_results["MOS_LOUD"],
    #             "MOS_NOISE": sigmos_results["MOS_NOISE"],
    #             "MOS_REVERB": sigmos_results["MOS_REVERB"],
    #             "MOS_SIG": sigmos_results["MOS_SIG"],
    #             "MOS_OVRL": sigmos_results["MOS_OVRL"],
    #         }
    #     ),
    #     ignore_index=True,
    # )

    # merge
    df = df_dnsmos
    df = pd.concat([df, df_nisqa], axis=1)
    df["MOS_SSL"] = df_mosssl
    # df = pd.concat([df, df_sigmos], axis=1)

    print(f'[len_in_sec] = {df["len_in_sec"][0]}') # not used
    print(f'OVRL_raw = {df["OVRL_raw"][0]}')
    print(f'SIG_raw = {df["SIG_raw"][0]}')
    print(f'BAK_raw = {df["BAK_raw"][0]}')
    print(f'OVRL = {df["OVRL"][0]}')
    print(f'SIG = {df["SIG"][0]}')
    print(f'BAK = {df["BAK"][0]}')
    print(f'P808_MOS = {df["P808_MOS"][0]}')
    print(f'mos_pred = {df["mos_pred"][0]}')
    print(f'noi_pred = {df["noi_pred"][0]}')
    print(f'dis_pred = {df["dis_pred"][0]}')
    print(f'col_pred = {df["col_pred"][0]}')
    print(f'loud_pred = {df["loud_pred"][0]}')
    # print(f'MOS_SSL = {df["MOS_SSL"][0]}')
    # print(f'MOS_COL = {df["MOS_COL"][0]}')
    # print(f'MOS_DISC = {df["MOS_DISC"][0]}')
    # print(f'MOS_LOUD = {df["MOS_LOUD"][0]}')
    # print(f'MOS_NOISE = {df["MOS_NOISE"][0]}')
    # print(f'MOS_REVERB = {df["MOS_REVERB"][0]}')
    # print(f'MOS_SIG = {df["MOS_SIG"][0]}')
    # print(f'MOS_OVRL = {df["MOS_OVRL"][0]}')
    print(f'[mean MOS] = {df.drop(["len_in_sec"], axis=1).iloc[0].mean(axis=0)}') # not used
    return df

if __name__ == "__main__":
    
    wavdir = "D:/Temp/sample_16k/sample_16k/传统"
    path_name = "传统"
    df = pd.DataFrame(
        columns=[
            "filename",
            "len_in_sec", # not used
            "OVRL_raw",
            "SIG_raw",
            "BAK_raw",
            "OVRL",
            "SIG",
            "BAK",
            "P808_MOS",
            "mos_pred",
            "noi_pred",
            "dis_pred",
            "col_pred",
            "loud_pred",
            # "MOS_SSL",
            # "MOS_COL",
            # "MOS_DISC",
            # "MOS_LOUD",
            # "MOS_NOISE",
            # "MOS_REVERB",
            # "MOS_SIG",
            # "MOS_OVRL",
        ]
    )
    count = 0
    for wav in glob.glob(os.path.join(wavdir, "*.wav")):
        sampling_rate = 16_000
        waveform = librosa.load(wav, sr=sampling_rate)[0]
        print(f"========== Processing Waveform No.{count} [{wav}] ==========")
        df_wav = process_waveform(waveform, sampling_rate,wav)
        df_wav["filename"] = wav
        df = df._append(df_wav, ignore_index=True)
        count += 1

    # 定义中英文列名对照
    column_names = {
        'filename': '文件名',
        'len_in_sec': '音频时长(秒)',
        'OVRL_raw': '原始整体评分',
        'SIG_raw': '原始信号质量',
        'BAK_raw': '原始背景质量', 
        'OVRL': '整体评分(MOS)',
        'SIG': '信号质量(MOS)',
        'BAK': '背景质量(MOS)',
        'P808_MOS': 'P.808标准MOS',
        'mos_pred': '预测MOS总分',
        'noi_pred': '噪声程度',
        'dis_pred': '断续程度', 
        'col_pred': '音色失真',
        'loud_pred': '音量适宜性',
        'MOS_SSL': 'SSL模型MOS'
    }

    # 重命名列名后输出CSV
    df.rename(columns=column_names, inplace=True)
    df.to_csv(path_name+".csv", index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持中文

    # 设置中文显示和样式（无需交互式后端）
    plt.switch_backend('agg')  # 非交互式后端，适合保存文件
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    

    # 创建输出目录（如果不存在）
    os.makedirs(path_name, exist_ok=True)

    # ==================================================================
    # 1. 核心指标对比图
    # ==================================================================
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df[['整体评分(MOS)', '信号质量(MOS)', '背景质量(MOS)']].melt(),
                x='variable', y='value', 
                estimator='mean', errorbar=('ci', 95),
                palette=["#FF9999", "#66B2FF", "#99FF99"])
    plt.title('语音质量核心指标均值对比（95%置信区间）')
    plt.xlabel('')
    plt.ylabel('MOS评分')
    plt.xticks(ticks=[0,1,2], labels=['整体质量', '信号质量', '背景质量'])
    plt.savefig(path_name+'/1_核心指标对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # 2. 问题维度雷达图
    # ==================================================================
    categories = ['噪声程度', '断续程度', '音量适宜性']
    values = df[categories].mean().values
    values = np.append(values, values[0])

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color='#FF6B6B', linewidth=2)
    ax.fill(angles, values, color='#FF6B6B', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('语音质量问题维度分析', pad=20)
    ax.set_rlabel_position(30)
    plt.yticks([1,2,3,4,5], ["1","2","3","4","5"], color="grey", size=8)
    plt.ylim(0,5)
    plt.savefig(path_name+'/2_问题维度雷达图.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # 3. 时长与质量关系图
    # ==================================================================
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(data=df, x='音频时长(秒)', y='整体评分(MOS)',
                            hue='噪声程度', size='断续程度',
                            sizes=(30, 200), palette='viridis',
                            alpha=0.7)
    plt.title('音频时长与质量关系\n(气泡大小=断续程度，颜色=噪声程度)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(path_name+'/3_时长质量关系图.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # 4. 各文件质量热力图
    # ==================================================================
    plt.figure(figsize=(10, 6))
    heatmap_data = df.set_index('文件名')[['整体评分(MOS)', '信号质量(MOS)', '背景质量(MOS)']]
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=.5, cbar_kws={'label': 'MOS评分'})
    plt.title('各音频文件质量评分热力图')
    plt.xlabel('质量维度')
    plt.ylabel('音频文件')
    plt.savefig(path_name+'/4_文件质量热力图.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("可视化结果已保存至 "+path_name+"/ 目录：")
    print(os.listdir(path_name))