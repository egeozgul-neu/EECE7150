import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class PointSelector:
    def __init__(self, center_img, other_img, other_name, num_points=6, out_dir="results"):
        self.num_points = num_points
        self.other_name = os.path.splitext(os.path.basename(other_name))[0]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Convert to RGB for display
        self.center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
        self.other_img = cv2.cvtColor(other_img, cv2.COLOR_BGR2RGB)

        # Combine the two images side by side
        self.h1, self.w1, _ = self.center_img.shape
        self.h2, self.w2, _ = self.other_img.shape
        self.h = max(self.h1, self.h2)
        self.combined = np.zeros((self.h, self.w1 + self.w2, 3), dtype=np.uint8)
        self.combined[:self.h1, :self.w1] = self.center_img
        self.combined[:self.h2, self.w1:self.w1+self.w2] = self.other_img

        # Store points
        self.pts_center = []
        self.pts_other = []

        # State: 0 = waiting for left click, 1 = waiting for right click
        self.click_state = 0

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.imshow(self.combined)
        self.ax.set_title("Click LEFT image point, then RIGHT image point")
        self.ax.axis("off")

        # Connect mouse + keyboard
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.onkey)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if self.click_state == 0 and x < self.w1:
            # Click on left (center) image
            self.pts_center.append((x, y))
            print(f"Picked LEFT point {len(self.pts_center)}: ({x:.1f}, {y:.1f})")
            self.click_state = 1
            self.redraw()

        elif self.click_state == 1 and x >= self.w1:
            # Click on right (other) image
            self.pts_other.append((x - self.w1, y))  # shift into right coords
            print(f"Picked RIGHT point {len(self.pts_other)}: ({x - self.w1:.1f}, {y:.1f})")
            self.click_state = 0
            self.redraw()

            # Stop if enough pairs
            if len(self.pts_center) >= self.num_points and len(self.pts_other) >= self.num_points:
                print("\n✅ All points selected!\n")
                plt.close(self.fig)
                self.save_results()

    def onkey(self, event):
        # Undo with Ctrl+Z
        if event.key == 'ctrl+z':
            if self.click_state == 1 and len(self.pts_center) > 0:
                removed = self.pts_center.pop()
                print(f"↩️ Undo: removed LEFT point {removed}")
                self.click_state = 0
            elif self.click_state == 0 and len(self.pts_other) > 0:
                removedR = self.pts_other.pop()
                removedL = self.pts_center.pop()
                print(f"↩️ Undo: removed pair LEFT {removedL}, RIGHT {removedR}")
            else:
                print("Nothing to undo")
            self.redraw()

        # Reset all points with "r"
        elif event.key == 'r':
            self.pts_center.clear()
            self.pts_other.clear()
            self.click_state = 0
            print("🔄 Reset: cleared all selected points for this image pair")
            self.redraw()

    def redraw(self):
        """Redraw all selected points and lines"""
        self.ax.clear()
        self.ax.imshow(self.combined)
        self.ax.axis("off")

        for i, (ptL, ptR) in enumerate(zip(self.pts_center, self.pts_other), start=1):
            xL, yL = ptL
            xR, yR = ptR[0] + self.w1, ptR[1]
            self.ax.plot([xL, xR], [yL, yR], 'r-')
            self.ax.plot(xL, yL, 'bo')
            self.ax.plot(xR, yR, 'bo')
            self.ax.text(xL, yL-5, f"{i}", color="yellow", fontsize=10, ha="center")
            self.ax.text(xR, yR-5, f"{i}", color="yellow", fontsize=10, ha="center")

        self.ax.set_title(f"Selected {len(self.pts_center)} / {self.num_points} pairs "
                          "(Ctrl+Z = undo, R = reset)")
        self.fig.canvas.draw()

    def save_results(self):
        """Save JSON + annotated image, print correspondence table"""
        # Save JSON
        data = {
            "image_pair": self.other_name,
            "points_center": [list(map(float, pt)) for pt in self.pts_center],
            "points_other": [list(map(float, pt)) for pt in self.pts_other]
        }
        json_path = os.path.join(self.out_dir, f"pairs_{self.other_name}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Saved point pairs to {json_path}")

        # Save annotated image
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(self.combined)
        ax.axis("off")
        ax.set_title(f"Point pairs for {self.other_name}")

        for i, (ptL, ptR) in enumerate(zip(self.pts_center, self.pts_other), start=1):
            xL, yL = ptL
            xR, yR = ptR[0] + self.w1, ptR[1]
            ax.plot([xL, xR], [yL, yR], 'r-')
            ax.plot(xL, yL, 'bo')
            ax.plot(xR, yR, 'bo')
            ax.text(xL, yL-5, f"{i}", color="yellow", fontsize=10, ha="center")
            ax.text(xR, yR-5, f"{i}", color="yellow", fontsize=10, ha="center")

        img_path = os.path.join(self.out_dir, f"pairs_{self.other_name}.png")
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Saved annotated image to {img_path}")

        # Print correspondence table
        print("=============================================")
        print(f"   Selected Point Correspondences ({self.other_name})")
        print("=============================================")
        print(f"{'Pair':<5} {'Center Image (x,y)':<25} {'Other Image (x,y)':<25}")
        print("---------------------------------------------")
        for i, (ptL, ptR) in enumerate(zip(self.pts_center, self.pts_other), start=1):
            print(f"{i:<5} ({ptL[0]:6.1f}, {ptL[1]:6.1f})    "
                  f"({ptR[0]:6.1f}, {ptR[1]:6.1f})")
        print("=============================================\n")

    def get_points(self):
        plt.show()
        return (np.array(self.pts_center, dtype=np.float32),
                np.array(self.pts_other, dtype=np.float32))


def pick_points(center_img, other_img, other_name, num_points=6):
    selector = PointSelector(center_img, other_img, other_name, num_points=num_points)
    return selector.get_points()


def main():
    center_path = "photos/center.jpg"
    other_paths = ["photos/img4.jpg", "photos/img5.jpg"]

    center_img = cv2.imread(center_path)
    if center_img is None:
        print("Error: could not load center image.")
        return

    for idx, path in enumerate(other_paths):
        other_img = cv2.imread(path)
        if other_img is None:
            print(f"Error: could not load {path}")
            continue

        print(f"\n==== Picking points for {path} ====")
        pts_center, pts_other = pick_points(center_img, other_img, path, num_points=6)


if __name__ == "__main__":
    main()

