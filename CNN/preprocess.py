import cv2


 ## TESTING STUFF HERE
def convert_normalized_to_pixel(y0, x0, y1, x1, width, height):

    x0_pixel = int(x0 * width)
    y0_pixel = int(y0 * height)
    x1_pixel = int(x1 * width)
    y1_pixel = int(y1 * height)

    return (x0_pixel, y0_pixel, x1_pixel, y1_pixel)


def draw_bounding_box_on_image(image_path, label_file_path):
    """Draw a bounding box on the image based on the bounding box info in the label file."""
    image = cv2.imread(image_path)

    with open(label_file_path, 'r') as file:
        data = file.read().strip().split()
        _, cx, cy, bw, bh = map(float, data)
        width, height = image.shape[1], image.shape[0]
        x1, y1, x2, y2 = convert_normalized_to_pixel(cx, cy, bw, bh, width, height)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(f"Bounding Box: ({x1}, {y1}), ({x2}, {y2})")

    # Show the image
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Paths
image_path = "../data/data/MS-ASL/frames/test/images/again1_frames/frame_0.png"
label_file_path = "../data/data/MS-ASL/frames/test/labels/again1_frames.txt"

# Process the image and draw the bounding box
draw_bounding_box_on_image(image_path, label_file_path)
