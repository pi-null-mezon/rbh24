import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm


emb = pickle.load(open("/2T_0/hackaton2/MakeExperiments/embedding_dict_v3.pickle", "rb"))

original = []
gen = []
anime_list = []
photo_list = []
watercolor_list = []
real_list = []

for k, v in emb.items():
    if 'original' in k:
        original.append(norm(v))
    elif 'anime' in k:
        anime_list.append(norm(v))
    elif 'photo' in k:
        photo_list.append(norm(v))
    elif 'watercolor' in k:
        watercolor_list.append(norm(v))
    else:
        real_list.append(norm(v))

fig, ax = plt.subplots(figsize=(11,5))

data = {
    'norm': original+anime_list+photo_list+watercolor_list+real_list,
    'type': 
    ['original' for _ in range(len(original))] +
    ['anime' for _ in range(len(anime_list))] +
    ['photo' for _ in range(len(photo_list))] +
    ['watercolor' for _ in range(len(watercolor_list))] +
    ['real' for _ in range(len(real_list))]
    ,
}

plt.grid(color = 'black', linewidth = 0.1)
g = sns.kdeplot(
   data=data, x="norm", hue="type",
    fill=True, common_norm=False,
   alpha=.4, linewidth=0,
)

plt.title("insightface/buffalo_l vs InstantID")
plt.savefig('test/result/buffalo_l_instantID.png')


anime_list, photo_list, watercolor_list, real_list = pickle.load(open("/2T_0/hackaton2/MakeExperiments/cos_sim_with_gen_img_v3.pickle", "rb"))

fig, ax = plt.subplots(figsize=(11,5))

data = {
    'cosine': 
    [i*2-1 for i in anime_list] + 
    [i*2-1 for i in photo_list] + 
    [i*2-1 for i in watercolor_list] + 
    [i*2-1 for i in real_list],
    'type': 
    ['anime' for _ in range(len(anime_list))] +
    ['photo' for _ in range(len(photo_list))] +
    ['watercolor' for _ in range(len(watercolor_list))] +
    ['real' for _ in range(len(real_list))]
    ,
}

plt.grid(color = 'black', linewidth = 0.1)
g = sns.kdeplot(
   data=data, x="cosine", hue="type",
    fill=True, common_norm=False,
   alpha=.4, linewidth=0,
)
ax.set_xlim(-1, 1)
plt.xticks([(i-10)*0.1 for i in range(21)] )

plt.title("insightface/buffalo_l on InstantID gen image")
plt.savefig('test/result/cosine_buffalo_l_instantID.png')
