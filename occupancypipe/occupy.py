import os
os.environ["LIBFREENECT2_LOGGER_LEVEL"] = "error"
from cycler import K
import open3d as o3d
from time import perf_counter, sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
from freenect2 import Device, FrameType
import random
import time
from scipy import ndimage

class Kinect:
    def __init__(self, inputdir=None):
        self.calibrated = False
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        if inputdir is not None:
            self.inputdir = inputdir
        else:
            self.inputdir = '/Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/occupancypipe'
        pass
    def loadPointCloud(self, filename, view=False):
        points = []
        with open(filename, 'r') as f:
            for line in f:
                coords = line.strip().split()
                if len(coords) == 3:
                    x, y, z = map(float, coords)
                    points.append([x, y, z])
        if view: self.loadPCTXT(filename)
        return np.array(points)

    def record(self, duration=5, fps=10):
        frame_interval = 1.0 / fps
        device = Device()
        frames = {}
        arr = []
        lidarFPS = 30
        frame_count = 0
        total_frames = duration * fps
        ret = []
        with device.running():
            start_time = perf_counter()
            print("------------------------------ Recording Started ------------------------------")
            while frame_count < duration * lidarFPS:
                type, frame = device.get_next_frame(timeout=1.0)
                # print(type)
                if type is FrameType.Depth:
                    frame_count += 1
                    if frame_count % (lidarFPS // fps) == 0:
                        points = device.registration.get_points_xyz_array(frame)
                        arr.append(points)
            print("------------------------------ Recording Ended ------------------------------")
        for points in arr:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
            pcd.remove_non_finite_points()
            ret.append(np.asarray(pcd.points))
                
        ret = np.array(ret, dtype=object)
        return ret

    def takeSingleFrame(self, view=False, save=False):
        device = Device()
        frames = {}
        with device.running():
            for type_, frame in device:
                frames[type_] = frame
                if FrameType.Depth in frames:
                    break

        depth = frames[FrameType.Depth]
        points = device.registration.get_points_xyz_array(depth)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd.remove_non_finite_points()
        npa = np.asarray(pcd.points)
        if save: self.saveNPY(npa)
        if view: o3d.visualization.draw_geometries([pcd])
        return npa
    
    def loadFrame(self, filename, type, view=False):
        if type == 'npy':
            points = np.load(filename)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.remove_non_finite_points()
            if view: o3d.visualization.draw_geometries([pcd])
            return np.asarray(pcd.points)
        elif type == 'pcd':
            pcd = o3d.io.read_point_cloud(filename)
            pcd.remove_non_finite_points()
            points = np.asarray(pcd.points)
            if view: o3d.visualization.draw_geometries([pcd])
            return points
        elif type == 'txt':
            data = np.loadtxt(filename)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data)
            if view: o3d.visualization.draw_geometries([pcd])
            points = np.asarray(pcd.points)
            return points
        else:
            raise ValueError("Unsupported file type. Use 'npy', 'pcd', or 'txt'.")
    
        

    def saveNPY(self, points, filename=None):
        output_dir = f"{self.inputdir}/frames"
        if filename is not None:
            output = os.path.join(output_dir, filename)
        else:
            output = os.path.join(output_dir, f"./frame_{random.randint(0, 10000000)}.npy")
        np.save(output, points)
        print(f"Saved frame to {output}")
        return output
    

    def saveVideo(self, frames, filename=None, fps=10, compress=True):
        output_dir = f"{self.inputdir}/videos"
        if filename is not None:
            output = os.path.join(output_dir, filename)
        else:
            output = os.path.join(output_dir, f"./video_{random.randint(0, 10000000)}.{'npz' if compress else 'npy' }")
        if compress:
            np.savez_compressed(output, frames=frames)
        else:
            np.save(output, frames)
        print(f"Saved video to {output}")
        return output

    def loadVideo(self, filename):
        frames = np.load(filename, allow_pickle=True)
        if filename.endswith('.npz'):
            frames = frames['frames']
        return frames
    
    def calibrate(self, points):
        self.x_min, self.x_max = points[:, 0].min(), points[:, 0].max()
        self.y_min, self.y_max = points[:, 1].min(), points[:, 1].max()
        self.calibrated = True
        
    def create(self, points, grid_resolution=0.02, z_min_threshold=1.45, z_max_threshold=1.80):
        table_points = points[(points[:, 2] >= z_min_threshold) & (points[:, 2] <= z_max_threshold)]
        # if len(table_points) == 0: return None, None
        if self.calibrated:
            x_min, x_max = self.x_min, self.x_max
            y_min, y_max = self.y_min, self.y_max
        else:
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

    def denoise(self, occupancy_grid, min_size=15 ):
        # this needs to be more efficient if its live
        # for every point, if none of its neighbors are occupied set it to unoccupied
        # for x in range(1, occupancy_grid.shape[0]-1):
        #     for y in range(1, occupancy_grid.shape[1]-1):
        #         if occupancy_grid[x, y] == 1:
        #             z = np.sum(occupancy_grid[x-1:x+2, y-1:y+2])
        #             if z <= 1:
        #                 occupancy_grid[x, y] = 0
        # return occupancy_grid
        mask = occupancy_grid == 1
        structure = ndimage.generate_binary_structure(2, 2)
        labels,n = ndimage.label(mask, structure=structure)
        counts = np.bincount(labels.ravel())
        remove = counts < min_size
        remove[0] = False
        occupancy_grid[remove[labels]] = 0
        return occupancy_grid
        
    def printgrid(self, occupancy_grid, extent, name):
        occupancy_grid = self.denoise(occupancy_grid)
        plt.figure(figsize=(10, 10))
        plt.imshow(occupancy_grid, cmap='binary', origin='upper', extent=extent)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('2D Occupancy Grid(Black is Occupied)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name}.png")
        plt.show()

    def persontest(self, save):
        points = self.loadPointCloud("person.txt", view=False)
        occupancy_grid, extent = self.create(
            points,
            grid_resolution=0.02,
            z_min_threshold=0.0,
            z_max_threshold=1.25
        )
        if save: np.save('person.npy', occupancy_grid)
        self.printgrid(occupancy_grid, extent, "Person")

    def tabletest(self, save):
        points = self.loadPointCloud("table.txt", view=False)
        occupancy_grid, extent = self.create(
            points,
            grid_resolution=0.02,
            z_min_threshold=-2,
            z_max_threshold=1
        )
        if save: np.save('table.npy', occupancy_grid)
        self.printgrid(occupancy_grid, extent, "Table")

    def frameto2d(self, save, points):
        occupancy_grid, extent = self.create(
            points,
            grid_resolution=0.02,
            z_min_threshold=-2,
            z_max_threshold=-0.5
        )
        if save: np.save('frame.npy', occupancy_grid)
        self.printgrid(occupancy_grid, extent, "2DfromFrame")
    
    def loadLiveFrame(self):
        # time.sleep(5)
        file = self.saveFrame(self.takeSingleFrame())
        points = self.loadNPY(file)
        
        # print(f"Total points: {len(points)}")
        # print(f"Points shape: {points.shape}")
        # print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        # print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        # print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        occupancy_grid, extent = self.create(
            points,
            grid_resolution=0.02,
            z_min_threshold=-1.5,
            z_max_threshold=-0.5
        )
        print(occupancy_grid.shape)
        self.printgrid(occupancy_grid, extent, "Live")
        
    def loadSavedFrame(self, filename, type='npy', view=False):
        points = self.loadFrame(filename, type=type, view=view)
        occupancy_grid, extent = self.create(
            points,
            grid_resolution=0.02,
            z_min_threshold=0.0,
            z_max_threshold=1.25
        )
        # printgrid(occupancy_grid, extent, "SavedFrame")
        return occupancy_grid, extent

    def createVideo(self, video):
        grid_resolution=0.02
        z_min_threshold=-1.720
        z_max_threshold=-0.5
        frames = []
        for i, points in enumerate(video):
            occupancy_grid, extent = self.create(
                points,
                grid_resolution=grid_resolution,
                z_min_threshold=z_min_threshold,
                z_max_threshold=z_max_threshold
            )
            frames.append(occupancy_grid)
            # printgrid(occupancy_grid, extent, f"VideoFrame_{i}")
        return frames, extent

    def videoPlayback(self, frames, extent):
        plt.ion()
        fig, ax = plt.subplots()
        img = plt.imshow(frames[0], cmap='binary', origin='upper', extent=extent)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('2D Occupancy Grid(Black is Occupied)')
        plt.tight_layout()
        fig.canvas.draw()
        for frame in frames[1:]:
            frame = self.denoise(frame)
            img.set_data(frame)
            fig.canvas.draw()
            plt.pause(0.1)
            
            # printgrid(occupancy_grid, extent, f"VideoFrame_{i}")
    def getZRange(self):
        points = self.takeSingleFrame(view=False)
        print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        return points
    # def zRangePlot(self, points):
        
        
    


if __name__ == "__main__":
    time.sleep(3)
    kinect = Kinect()
    # video = kinect.loadVideo("/Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/occupancypipe/videos/video_4923851.npy")
    # frame = kinect.loadNPY("/Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/occupancypipe/frame_9059672.npy", view=False)
    # print("Calibrating Kinect...")
    # kinect.frameto2d(save=False, points=frame)
    # frame = kinect.takeSingleFrame()
    # kinect.frameto2d(save=False, points=frame)
    # kinect.saveNPY(frame, filename="calibration_frame.npy")
    
    frame = kinect.loadFrame("occupancypipe/frames/calibration_frame.npy", type='npy', view=False)
    kinect.calibrate(frame)

    
    
    #     # print("Taking Frame...")
    # # kinect.saveFrame(frame)
    # video = kinect.record()
    # filename = kinect.saveVideo(video, compress=True)
    video = kinect.loadVideo("occupancypipe/videos/video_9489441.npy")
    filename = kinect.saveVideo(video, compress=True)
    video = kinect.loadVideo(filename=filename)
    
    frames, extent = kinect.createVideo(video)
    kinect.videoPlayback(frames, extent=extent)
    # print(len(video))
    # out = saveVideo(video)x
    # video = loadVideo(out)
    # print(len(video))