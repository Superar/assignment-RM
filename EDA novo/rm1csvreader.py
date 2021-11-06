from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import os
import math
sep = os.path.sep
matplotlib.use('TkAgg')

#init vars
clean_dataset = False
clean_dataset_nocutoff = False
show_avg_surface = True
show_std_surface = True
eda = True
debug = False
save_figure = True
show_plts = True # nao mexer nisto xD

nprobs = 11
probs = [0.02 * math.sqrt(2) **i for i in range(nprobs)]
exam_gap = 4
max_exams = 100
exams = [2] + [i for i in range(exam_gap, max_exams + exam_gap, exam_gap)]
#exams[-1] = 299
nexams = len(exams)
seeds = 5
line_start = 1
filename = "results_11_06_2021_05_25_49.csv"
clean_output = filename.split(".")[0] + "_clean.csv"
clean_output_nocutoff = filename.split(".")[0] + "_clean_nocutoff.csv"
plots_dir = "EDA" + sep + "" + sep

# matrix intialization
code1_time = np.zeros((nexams, nprobs, seeds), dtype=np.float)
code2_time = np.zeros((nexams, nprobs, seeds), dtype=np.float)
code1_n = np.zeros((nexams, nprobs, seeds), dtype=np.int)
code2_n = np.zeros((nexams, nprobs, seeds), dtype=np.int)
code1_c = np.zeros((nexams, nprobs), dtype=np.int)
code2_c = np.zeros((nexams, nprobs), dtype=np.int)
probs_map = dict(zip(probs, range(nprobs)))
exams_map = dict(zip(exams, range(nexams)))

# useful for EDA
if clean_dataset:
    lc = 0
    with open(clean_output, 'w', newline='') as fd:
        with open(filename, 'r', newline='') as ff:
            writer = csv.writer(fd, delimiter=',')
            reader = csv.reader(ff, delimiter=',')
            for row in reader:
                if lc >= line_start:
                    temp = row[1].replace(".in", "").split("_")[1].split("-")
                    writer.writerow([temp[0], temp[1], temp[2], row[2], row[3], row[4]])
                else:
                    writer.writerow(["Exams", "Probability", "Seed", "Code Type", "N", "Time"])
                lc += 1
if clean_dataset_nocutoff:
    lc = 0
    with open(clean_output_nocutoff, 'w', newline='') as fd:
        with open(filename, 'r', newline='') as ff:
            writer = csv.writer(fd, delimiter=',')
            reader = csv.reader(ff, delimiter=',')
            for row in reader:
                if lc >= line_start:
                    temp = row[1].replace(".in", "").split("_")[1].split("-")
                    if int(row[3]) != -1:
                        writer.writerow([temp[0], temp[1], temp[2], row[2], row[3], row[4]])
                else:
                    writer.writerow(["Exams", "Probability", "Seed", "Code Type", "N", "Time"])
                lc += 1

def plt_manage(fname = "image.png", sb_fig = None):
    if save_figure:
        if sb_fig is None:
            plt.savefig(fname)
        else:
            fig = sb_fig
            fig.savefig(fname)
        print("Saved " + fname + ".")
    if show_plts:
        plt.show()


print("\nSaving figures...")
if eda:
    pair_plt = True
    corr_plt = True
    box_plt = True
    n_hist_plot = True
    time_hist_plot = True
    examtime_plt = True
    probtime_plt = True
    ntime_plt = True
    code_type = True

    #datasetf = clean_output_nocutoff
    datasetf = clean_output
    eda_plots_dir = "EDA" + sep + ("standard_dataset" if datasetf is clean_output else "no_cutoffs_dataset") + sep

    df = pd.read_csv(datasetf)
    print("\nData sample:")
    print(df.head())
    print("\nData description: ")
    print(df.describe())
    print("\nData types:")
    print(df.dtypes)
    print("\nData info:")
    df.info()

    if pair_plt:
        f = sns.pairplot(df)
        plt_manage(eda_plots_dir + "pairplot.png", f)

    if examtime_plt:
        #fig = plt.figure()
        sns.scatterplot(x='Exams', y='Time', data=df, hue='Time')
        #plt.savefig(eda_plots_dir + 'time_by_exam.png')
        #plt.close(fig)
        plt_manage(eda_plots_dir + "time_by_exam.png")
    if probtime_plt:
        #fig = plt.figure()
        sns.scatterplot(x='Probability', y='Time', data=df, hue='Time')
        plt_manage(eda_plots_dir + "time_by_prob.png")
        #plt.savefig(eda_plots_dir + 'time_by_prob.png')
        #plt.close(fig)
    if ntime_plt:
        sns.scatterplot(x='N', y='Time', data=df, hue='Time')
        plt_manage(eda_plots_dir + "time_by_n.png")
    if code_type:
        sns.scatterplot(x='Code Type', y='Time', data=df, hue='Time')
        plt_manage(eda_plots_dir + "time_by_code.png")

    if n_hist_plot:
        df["N"].plot(kind="hist", orientation='horizontal', cumulative=False)
        plt_manage(eda_plots_dir + "n_hist.png")
    if time_hist_plot:
        df["Time"].plot(kind="hist", orientation='horizontal', cumulative=False)
        plt_manage(eda_plots_dir + "time_hist.png")

    if box_plt:
        color = dict(boxes='orangered', whiskers='orangered', medians = 'blue', caps = 'black')
        df[["N", "Time"]].plot(kind="box", subplots = True, layout = (1, 2),
                               sharex = False, sharey = False, figsize = (16, 8), color = color, sym='r+')
        plt_manage(eda_plots_dir + "boxplot.png")

    if corr_plt:
        mask = np.tril(df.corr())
        sns.heatmap(df.corr(), fmt=".1g", annot = True, cmap = "cool", mask = mask)
        plt_manage(eda_plots_dir + "correlation_plot.png")


lc = 0
with open(filename, 'r', newline='') as fd:
    csv_reader = csv.reader(fd, delimiter=',')
    for row in csv_reader:
        if lc >= line_start:
            temp = row[1].replace(".in", "").split("_")[1].split("-")
            exam_ind = exams_map[int(temp[0])]
            prob_ind = probs_map[float(temp[1])]
            if row[2] == "code1":
                cnt = code1_c[exam_ind][prob_ind]
                code1_n[exam_ind][prob_ind][cnt] = int(row[3])
                code1_time[exam_ind][prob_ind][cnt] = float(row[4])
                code1_c[exam_ind][prob_ind] += 1
            else:
                cnt = code2_c[exam_ind][prob_ind]
                code2_n[exam_ind][prob_ind][cnt] = int(row[3])
                code2_time[exam_ind][prob_ind][cnt] = float(row[4])
                code2_c[exam_ind][prob_ind] += 1
        lc += 1



code1_mean_time = np.mean(code1_time, axis=2)
code2_mean_time = np.mean(code2_time, axis=2)
code1_std_time = np.std(code1_time, axis=2)
code2_std_time = np.std(code2_time, axis=2)
xv, yv = np.meshgrid(np.log2(np.multiply(probs, 100)), exams)


if debug:
    print("Code 1 time:")
    print(code1_time)
    print("Code 1 shape:")
    print(code1_mean_time.shape)
    print("X matrix:")
    print(xv)
    print("Y matrix:")
    print(yv)


if show_avg_surface:
    zv1 = code1_mean_time
    zv2 = code2_mean_time

    fig = plt.figure(figsize=(16, 8))
    #fig = plt.figure(figsize=plt.figaspect(0.5))

    def format_func(val, tick_number):
        return r"${0}$".format(round(0.02 * 2 ** (val - 1), 3))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xv, yv, zv1, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    #ax.plot3D(xline, yline, c=zdata, cmap='Greens')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_xlabel('Log of exam probability')
    ax.set_ylabel('Number of exams')
    ax.set_zlabel('Time (sec)')
    ax.set_title("Average timings (code1.c)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(xv, yv, zv2, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    #ax.plot3D(xline, yline, c=zdata, cmap='Greens')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_xlabel('Log of exam probability')
    ax.set_ylabel('Number of exams')
    ax.set_zlabel('Time (sec)')
    ax.set_title("Average timings (code2.c)")

    plt_manage("avg_timings.png")



if show_std_surface:
    fig = plt.figure(figsize=(16, 8))
    #fig = plt.figure(figsize=plt.figaspect(0.5))

    zv1std = code1_std_time
    zv2std = code2_std_time

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xv, yv, zv1std, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    #ax.plot3D(xline, yline, c=zdata, cmap='Greens')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_xlabel('Log of exam probability')
    ax.set_ylabel('Number of exams')
    ax.set_zlabel('Time (sec)')
    ax.set_title("Standard timings (code1.c)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(xv, yv, zv2std, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    #ax.plot3D(xline, yline, c=zdata, cmap='Greens')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_xlabel('Log of exam probability')
    ax.set_ylabel('Number of exams')
    ax.set_zlabel('Time (sec)')
    ax.set_title("Standard timings (code2.c)")

    plt_manage("std_timings.png")