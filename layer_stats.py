from glob import glob
from PIL import Image

# Get all statistics png files
stat_pngs = glob("*_placement_iterations/overlap_statistics.png")

# Turn all of these into pillow images
stat_images = [Image.open(png) for png in stat_pngs]

# Set all their alpha values to 1/len(stat_images)
for image in stat_images:
    image.putalpha(255 // len(stat_images))

# Merge all of these images into one
for i in range(1, len(stat_images)):
    stat_images[0] = Image.alpha_composite(stat_images[0], stat_images[i])

stat_images[0].show()