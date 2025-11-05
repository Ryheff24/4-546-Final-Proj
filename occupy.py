import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def loadPointCloud(filename, view=False):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            coords = line.strip().split()
            if len(coords) == 3:
                x, y, z = map(float, coords)
                points.append([x, y, z])
    if view: viewPointCloud(filename)
    return np.array(points)

def viewPointCloud(filename):
    data = np.loadtxt(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])
    
def create(points, grid_resolution=0.02, z_min_threshold=1.45, z_max_threshold=1.80):
    table_points = points[(points[:, 2] >= z_min_threshold) & (points[:, 2] <= z_max_threshold)]
    if len(table_points) == 0: return None, None
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    grid_width = int(np.ceil((x_max - x_min) / grid_resolution))
    grid_height = int(np.ceil((y_max - y_min) / grid_resolution))
    occupancy_grid = np.zeros((grid_height, grid_width))
    for point in table_points:
        x, y, _ = point
        
        grid_x = int((x - x_min) / grid_resolution)
        grid_y = int((y - y_min) / grid_resolution)
        grid_x = max(0, min(grid_x, grid_width - 1))
        grid_y = max(0, min(grid_y, grid_height - 1))
        occupancy_grid[grid_y, grid_x] = 1
    
    return occupancy_grid, (x_min, x_max, y_min, y_max)


def printgrid(occupancy_grid, extent, name):
    plt.figure(figsize=(10, 10))
    plt.imshow(occupancy_grid, cmap='binary', origin='lower', extent=extent)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('2D Occupancy Grid(Black is Occupied)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()

def main():
    points = loadPointCloud("person.txt", view=False)
    save = False
    occupancy_grid, extent = create(
        points,
        grid_resolution=0.02,
        z_min_threshold=0.0,
        z_max_threshold=1.25
    )

    if save: np.save('person.npy', occupancy_grid)

    printgrid(occupancy_grid, extent, "Person")

    # z_min_threshold=1.1,
    # z_max_threshold=1.5
    points = loadPointCloud("table.txt", view=False)
    save = False
    occupancy_grid, extent = create(
        points,
        grid_resolution=0.02,
        z_min_threshold=1.1,
        z_max_threshold=1.5
    )


    printgrid(occupancy_grid, extent, "Table")
    
    if save: np.save('table.npy', occupancy_grid)
    
if __name__ == "__main__":
    main()
