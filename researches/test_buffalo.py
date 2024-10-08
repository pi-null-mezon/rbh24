import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm


def compute_sim(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim
    

new_pair1, embedding = pickle.load(open("/2T_0/hackaton2/MakeExperiments/open_test_embedding.pickle", "rb"))
new_pair2, embedding2 = pickle.load(open("/2T_0/hackaton2/MakeExperiments/geofaces_embedding.pickle", "rb"))

new_pair = list(new_pair1.items()) + list(new_pair2.items())
embedding.update(embedding2)

same_pair = []
diff_pair = []

for key, value in new_pair:
    emb1 = embedding[value[0]]
    emb2 = embedding[value[1]]
    same_sim = compute_sim(emb1, emb2)
    same_pair.append(same_sim)
    
    for diff_key, diff_value in new_pair:
        if diff_key != key:
            emb3 = embedding[diff_value[0]]
            emb4 = embedding[diff_value[1]]
            diff_pair.append(compute_sim(emb1, emb3))
            diff_pair.append(compute_sim(emb1, emb4))
            diff_pair.append(compute_sim(emb2, emb3))
            diff_pair.append(compute_sim(emb2, emb4))
    
 
fig, ax = plt.subplots(figsize=(11,5))

g = sns.kdeplot(
   data={'cos_sim': same_pair+diff_pair, 'type': ['same' for _ in range(len(same_pair))] + ['diff' for _ in range(len(diff_pair))]}, x="cos_sim", hue="type",
   fill=True, common_norm=False, 
   alpha=.4, linewidth=0,
)
ax.set_xlim(0, 1)
ax.set_xlabel('cosine')
plt.xticks([(i-10)*0.1 for i in range(21)] )
plt.title("Test set, 1849-id")
plt.grid(color = 'black', linewidth = 0.1)

plt.legend(title='Type verification', loc='upper left', labels=['diff ', 'same '])
plt.savefig('test/result/cos_sim_dist.png')

same_sim = np.array(same_pair)
diff_sim = np.array(diff_pair)

thresholds = np.arange(-1.0, 1.0, 0.0001)

cos_same = np.array(same_sim)
same_total = same_sim.shape[0]
cos_diff = np.array(diff_sim)
diff_total = diff_sim.shape[0]
fnmr, fmr = [], []
for t in thresholds:
    tmp_fnmr = np.sum(cos_same < t) / same_total
    tmp_fmr = np.sum(cos_diff >= t) / diff_total
    fnmr.append(tmp_fnmr)
    fmr.append(tmp_fmr)

closest_key = np.argmin([abs(i - 1e-6) for i in fmr])
print(fmr[closest_key])
print(fnmr[closest_key])

fig, ax = plt.subplots(figsize=(11,5))

data = {
    'threshold': thresholds.tolist()+thresholds.tolist(),
    'fnmr_fmr': fmr+fnmr,
    'type': ['fmr' for _ in range(len(fmr))] + ['fnmr' for _ in range(len(fnmr))],
}

g = sns.lineplot(data=data, x="threshold", y="fnmr_fmr", hue='type')

ax.set_xlim(-1, 1)
ax.set_xlabel('cosine')
plt.xticks([(i-10)*0.1 for i in range(21)] )
plt.title("Test set, 1849-id")
plt.grid(color = 'black', linewidth = 0.1)

ax.axvline(fnmr[closest_key], color='r')

plt.savefig('test/result/fnmr_fmr.png')
