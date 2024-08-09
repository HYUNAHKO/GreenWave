import cv2
import os

def apply_clahe(image, clip_limit=20.0, tile_grid_size=(2, 2)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def merge_channels_with_clahe(red_channel_file, green_channel_file, blue_channel_file, output_tif_file, output_png_file):

    r_channel = cv2.imread(red_channel_file, cv2.IMREAD_GRAYSCALE)
    g_channel = cv2.imread(green_channel_file, cv2.IMREAD_GRAYSCALE)
    b_channel = cv2.imread(blue_channel_file, cv2.IMREAD_GRAYSCALE)

    if r_channel is None or g_channel is None or b_channel is None:
        print("Error: One or more input files could not be read.")
        return

    r_channel = apply_clahe(r_channel, clip_limit=20.0, tile_grid_size=(2, 2))
    g_channel = apply_clahe(g_channel, clip_limit=20.0, tile_grid_size=(2, 2))
    b_channel = apply_clahe(b_channel, clip_limit=20.0, tile_grid_size=(2, 2))

    merged_img = cv2.merge([b_channel, g_channel, r_channel])

    # Ensure the output directories exist
    output_dir_tif = os.path.dirname(output_tif_file)
    if not os.path.exists(output_dir_tif):
        os.makedirs(output_dir_tif)
    
    output_dir_png = os.path.dirname(output_png_file)
    if not os.path.exists(output_dir_png):
        os.makedirs(output_dir_png)

    cv2.imwrite(output_tif_file, merged_img)
    cv2.imwrite(output_png_file, merged_img)

def split_image(image_path, output_dir, grid_size=(24, 24)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image file {image_path} could not be read.")
        return

    img_height, img_width, _ = img.shape

    tile_height = img_height // grid_size[1]
    tile_width = img_width // grid_size[0]

    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            tile = img[y:y + tile_height, x:x + tile_width]
            tile_filename = os.path.join(output_dir, f'tile_{x}_{y}.png')
            cv2.imwrite(tile_filename, tile)

# File paths
red_channel_file = r'/Users/caz/Downloads/Clipping/K3A_20200504045549_28203_00042683_L1R_PS-002/K3A_20200504045549_28203_00042683_L1R_PR.tif'
green_channel_file = r'/Users/caz/Downloads/Clipping/K3A_20200504045549_28203_00042683_L1R_PS-002/K3A_20200504045549_28203_00042683_L1R_PG.tif'
blue_channel_file = r'/Users/caz/Downloads/Clipping/K3A_20200504045549_28203_00042683_L1R_PS-002/K3A_20200504045549_28203_00042683_L1R_PB.tif'
output_tif_file = r'/Users/caz/Downloads/Clipping/K3A_20200504045549_28203_00042683_L1R_PS-002/output_image.tif'
output_png_file = r'/Users/caz/Downloads/Clipping/K3A_20200504045549_28203_00042683_L1R_PS-002/output_image.png'
output_dir_cut = r'/Users/caz/Downloads/Clipping/K3A_20200504045549_28203_00042683_L1R_PS-002/output_tiles'

# Ensure the output directory for the cut images exists
if not os.path.exists(output_dir_cut):
    os.makedirs(output_dir_cut)

# Merge channels with CLAHE
merge_channels_with_clahe(red_channel_file, green_channel_file, blue_channel_file, output_tif_file, output_png_file)

# Split the merged image into tiles
split_image(output_png_file, output_dir_cut, grid_size=(24, 24))

print(f"RGB merged image with CLAHE saved to: {output_png_file}")
print(f"Image split into 24x24 tiles and saved to directory: {output_dir_cut}")
