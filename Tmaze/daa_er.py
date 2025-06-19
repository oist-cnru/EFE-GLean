#!/usr/bin/env python
import numpy as np
import os, argparse
import matplotlib.pyplot as plt

path = "../LibPvrnn/results/2d_pftagentdsg"
outpath = "./fixedrng/analysis"

## Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--subdir", help="subdir containing ER NPZ files", default="exp1_aifg")
parser.add_argument("--er_samples", "-p", help="Number of samples", type=int, default=100)
parser.add_argument("--sense_step", "-s", help="which sensory step to visualize", type=int, default=1)
args = parser.parse_args()

samples = args.er_samples
step = args.sense_step-1
tlen = 3
future_step = step
subdir = args.subdir

data = []
labels = []

data_x0 = []
data_x1 = []
data_x2 = []
data_d = []
data_muq = []
data_mup = []
data_recerr = []
data_kld = []

for idx in range(samples):
    dataz = np.load(os.path.join(path, subdir, "pgstep" + str(step) + ".npz"))
    data_x0.append(dataz["itr99_s" + str(idx) + "_output0_x"][:tlen,:])
    data_x1.append(dataz["itr99_s" + str(idx) + "_output1_x"][:tlen,:])
    data_x2.append(dataz["itr99_s" + str(idx) + "_output2_x"][:tlen,:])
    data_d.append(dataz["itr99_s" + str(idx) + "_layer1_z"][:tlen,:])
    data_muq.append(dataz["itr99_s" + str(idx) + "_layer1_muq"][:tlen,:])
    data_mup.append(dataz["itr99_s" + str(idx) + "_layer1_mup"][:tlen,:])
    data_recerr.append(dataz["itr99_s" + str(idx) + "_output0_recErr"][:tlen]+dataz["itr99_s" + str(idx) + "_output1_recErr"][:tlen]+dataz["itr99_s" + str(idx) + "_output2_recErr"][:tlen])
    data_kld.append(dataz["itr99_s" + str(idx) + "_layer1_kld"][:tlen])

data.append(np.array(data_x0))
labels.append(["Center", "Down", "Top left", "Top right"])
data.append(np.array(data_x1))
labels.append(["Blue", "Green", "Red", "None"])
data.append(np.array(data_x2))
labels.append(["Goal reached", "Goal not reached"])
# data.append(np.array(data_d))
# labels.append(None)
data.append(np.array(data_muq))
labels.append(None)
# data.append(np.array(data_mup))
# labels.append(None)

if not os.path.exists(os.path.join(outpath, subdir)):
    os.makedirs(os.path.join(outpath, subdir))

font = {"size": 30}
plt.rc("font", **font)
# Plot samples
for idx in range(samples):
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(5*2,9*2))
    for i,d in enumerate(data):
        axs[i].imshow(d[idx].T, aspect="auto")

    for i,a in enumerate(axs):
        a.set_xticks([i for i in range(data[i][0].shape[0])], minor=False)
        a.set_yticks([i for i in range(data[i][0].shape[1])], labels=labels[i], minor=False)
        a.axvline(x=future_step+0.5, linewidth=2, color='r')
    evidence_loss = np.sum(data_recerr[idx][:future_step]) + np.sum(data_kld[idx][:future_step])

    fig.suptitle("Sample " + str(idx))
    axs[0].set_ylabel("Position")
    axs[1].set_ylabel("Observation")
    axs[3].set_ylabel(r"$\mu^q$")
    axs[-1].set_xlabel("Timesteps")

    fig.tight_layout()
    print("Saved", os.path.join(outpath, subdir, "s" + str(idx) + ".png"))
    fig.savefig(os.path.join(outpath, subdir, "s" + str(idx) + ".png"))
    plt.close()

# Plot loss
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,9))
axs[0].imshow(np.array(data_recerr), aspect="auto")
axs[0].set_xticks([i for i in range(tlen)], minor=False)
axs[0].set_yticks([i for i in range(samples)], minor=False)
axs[0].axvline(x=future_step-0.5, linewidth=2, color='r')
axs[0].set_ylabel("Rec loss")
axs[1].imshow(np.array(data_kld), aspect="auto")
axs[1].set_xticks([i for i in range(tlen)], minor=False)
axs[1].set_yticks([i for i in range(samples)], minor=False)
axs[1].set_xlim(-0.5, tlen-0.5)
axs[1].axvline(x=future_step-0.5, linewidth=2, color='r')
axs[1].set_ylabel("KLD")
axs[1].set_xlabel("Timesteps")

fig.tight_layout()
fig.savefig(os.path.join(outpath, subdir, "loss.png"))
np.savetxt(os.path.join(outpath, subdir, "rec_err.txt"), np.array(data_recerr), delimiter=',')
np.savetxt(os.path.join(outpath, subdir, "kld.txt"), np.array(data_kld), delimiter=',')
