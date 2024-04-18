import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i in range(10):
    img_path = "./HW2/assets/random_exploration_{}.png".format(int(100/10)*(i))
    # print(int(93/100)*(i+1))
    img = mpimg.imread(img_path)
    ax = axs[i//5, i%5]
    ax.imshow(img)
    ax.axis('off')  

plt.tight_layout()
plt.show()