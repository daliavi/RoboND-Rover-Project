import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def color_thresh_all(img, min_thresh=(160, 160, 160), max_thresh=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > min_thresh[0]) & (img[:, :, 0] < max_thresh[0]) \
                   & (img[:, :, 1] > min_thresh[1]) & (img[:, :, 1] < max_thresh[1]) \
                   & (img[:, :, 2] > min_thresh[2]) & (img[:, :, 2] < max_thresh[2])

    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    return color_select


# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    bl, br, tl, tr = [14, 140], [301, 140], [200, 96], [118, 96]

    source = np.float32([bl, br, tl, tr])
    destination = np.float32([[img.shape[1] / 2 - dst_size, img.shape[0] - bottom_offset],
                              [img.shape[1] / 2 + dst_size, img.shape[0] - bottom_offset],
                              [img.shape[1] / 2 + dst_size, img.shape[0] - 2 * dst_size - bottom_offset],
                              [img.shape[1] / 2 - dst_size, img.shape[0] - 2 * dst_size - bottom_offset],
                              ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed_nav = color_thresh_all(warped, (160, 160, 160), (256, 256, 256))
    threshed_rock = color_thresh_all(warped, (120, 109, -1), (218, 198, 106))
    threshed_obs = color_thresh_all(warped, (-1, -1, -1), (106, 101, 97))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = threshed_obs * 255
    Rover.vision_image[:, :, 1] = threshed_rock * 255
    Rover.vision_image[:, :, 2] = threshed_nav * 255

    # 5) Convert map image pixel values to rover-centric coords
    xpix_nav, ypix_nav = rover_coords(threshed_nav)
    xpix_rock, ypix_rock = rover_coords(threshed_rock)
    xpix_obs, ypix_obs = rover_coords(threshed_obs)

    # 6) Convert rover-centric pixel values to world coordinates
    rover_yaw = Rover.yaw
    rover_xpos, rover_ypos = Rover.pos
    scale = 10

    x_world_nav, y_world_nav = pix_to_world(xpix_nav, ypix_nav, rover_xpos,
                                            rover_ypos, rover_yaw,
                                            Rover.worldmap.shape[0], scale)
    x_world_rock, y_world_rock = pix_to_world(xpix_rock, ypix_rock, rover_xpos,
                                              rover_ypos, rover_yaw,
                                              Rover.worldmap.shape[0], scale)
    x_world_obs, y_world_obs = pix_to_world(xpix_obs, ypix_obs, rover_xpos,
                                            rover_ypos, rover_yaw,
                                            Rover.worldmap.shape[0], scale)


    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_world_obs, x_world_obs, 0] += 1
    Rover.worldmap[y_world_rock, x_world_rock, 1] += 1
    Rover.worldmap[y_world_nav, x_world_nav, 2] += 1


    # 8) Convert rover-centric pixel positions to polar coordinates
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_nav, ypix_nav)
    
    return Rover