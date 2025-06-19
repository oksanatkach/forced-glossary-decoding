import torch
import math
import matplotlib.pyplot as plt

inputs = torch.load('next_token_logits.t')
print(torch.median(inputs))
print(inputs[24818])
print(inputs[5156])
inputs = inputs.detach().numpy()
inputs = inputs[inputs != -math.inf]
# print(inputs)


fig, ax = plt.subplots(1, 2)
# ax.hist(inputs, color='lightgreen', ec='black', bins=1000)
ax[0].hist(inputs, ec='lightblue', bins=1000)
ax[0].set_xlabel('word')
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_visible(False)
ax[0].set_title('Word')
ax[0].axvline(inputs[24818], color='k', linestyle='dashed', linewidth=1)
# ax[1].hist(inputs, color='lightgreen', ec='black', bins=1000)

plt.show()
