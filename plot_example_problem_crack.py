import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------

# output path

# -----------------------------------

out_dir = r"C:\Users\13144\Documents\Masters_Thesis\datasets\SUT_Compressed"
out_file = os.path.join(out_dir, "image_crop_windy_example.png")

# -----------------------------------

# parameters

# -----------------------------------

margin = 20

# endpoints

start = np.array([50, 100])
end   = np.array([250, 100])

# -----------------------------------

# generate windy crack

# -----------------------------------

x = np.linspace(start[0], end[0], 400)
y = 100 + 60 * np.sin((x - start[0]) * 0.05)

# -----------------------------------

# image_crop bounds (same logic)

# -----------------------------------

x_bound1 = max(min(start[0], end[0]) - margin, 0)
x_bound2 = max(max(start[0], end[0]) + margin, 0)

y_bound1 = max(min(start[1], end[1]) - margin, 0)
y_bound2 = max(max(start[1], end[1]) + margin, 0)

# detect points outside crop

outside = (x < x_bound1) | (x > x_bound2) | (y < y_bound1) | (y > y_bound2)

# -----------------------------------

# plot

# -----------------------------------

plt.figure(figsize=(8,6))

plt.plot(x, y, linewidth=2, label="Windy crack")
plt.scatter(x[outside], y[outside], color="red", s=10, label="Outside crop")

plt.scatter(*start, color="green", s=100, label="Start")
plt.scatter(*end, color="blue", s=100, label="End")

# crop rectangle

rect_x = [x_bound1, x_bound2, x_bound2, x_bound1, x_bound1]
rect_y = [y_bound1, y_bound1, y_bound2, y_bound2, y_bound1]
plt.plot(rect_x, rect_y, "k--", linewidth=2, label="Crop box")

plt.gca().invert_yaxis()
plt.legend()
plt.title("image_crop bounding box vs winding crack")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.savefig(out_file, dpi=300)
plt.close()

print("Saved figure to:", out_file)
