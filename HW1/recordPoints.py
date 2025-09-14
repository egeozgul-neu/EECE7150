import matplotlib.pyplot as plt
import cv2

# Load image
img = cv2.imread("HW1_image1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Global list to store clicks
points = []

# Callback function
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        print(f"Point selected: {(x, y)}")

        # Draw a red dot
        plt.plot(x, y, 'ro')
        plt.draw()

# Show image
fig, ax = plt.subplots()
ax.imshow(img_rgb)
ax.set_title("Click points (close window when done)")

# Connect callback
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

print("Final points:", points)

