from calendar import c
from fileinput import filename
import os

from sympy import N, comp
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
from queue import Queue
from freenect2 import Device, FrameType
import freenect2

# Fixing the queue of hte freenect2 library to avoid blocking on full queue
class FixedQueueFrameListener:
    def __init__(self, maxsize=16):
        self.queue = Queue(maxsize=maxsize)
    
    def __call__(self, frame_type, frame):
        if self.queue.qsize() >= self.queue.maxsize - 2:
            try:
                self.queue.get_nowait()
            except:
                pass
        try:
            self.queue.put_nowait((frame_type, frame))
        except:
            pass  
    
    def get(self, timeout=None):
        return self.queue.get(True, timeout)
freenect2.QueueFrameListener = FixedQueueFrameListener

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


    def record(self, duration=10, fps=10, playback=False):
        device = Device()
        arr = []
        lidarFPS = 30
        
        pcd = o3d.geometry.PointCloud()
        
        if playback:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Kinect Live playback', width=800, height=600)
            option = vis.get_render_option()
            option.background_color = np.asarray([0.1, 0.1, 0.1])
            first_frame = True

        with device.running():
            start_time = perf_counter()
            print("------------------------------ Recording Started ------------------------------")
            
            frame_count = 0
            target_frames = duration * lidarFPS
            
            for type_, frame in device:
                if type_ == FrameType.Depth:
                    # Convert to point cloud INSIDE the context while frame is valid
                    points = device.registration.get_points_xyz_array(frame)
                    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
                    pcd.remove_non_finite_points()
                    
                    # Make a copy of the points array
                    arr.append(np.asarray(pcd.points).copy())
                    
                    if playback:
                        if first_frame:
                            vis.add_geometry(pcd)
                            first_frame = False
                        else:
                            vis.update_geometry(pcd)
                        if not vis.poll_events():
                            break
                        vis.update_renderer()
                    
                    frame_count += 1
                    if frame_count >= target_frames:
                        break
            
            print("------------------------------ Recording Ended ------------------------------")
        
        if playback:
            vis.destroy_window()
        
        ret = np.array(arr, dtype=object)
        return ret
    def frameSkip(self, video, skip=2):
        return video[::skip]
    
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
    
    def create(self, points, grid_resolution=0.02, z_min_threshold=1.45, z_max_threshold=1.80, skipCalibration=False):
        table_points = points[(points[:, 2] >= z_min_threshold) & (points[:, 2] <= z_max_threshold)]
        # if len(table_points) == 0: return None, None
        if self.calibrated and not skipCalibration:
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

    def createVideo(self, video, skipCalibration=False, grid_resolution=0.02, z_min_threshold=-1.9, z_max_threshold=-0.5):
        frames = []
        for i, points in enumerate(video):
            occupancy_grid, extent = self.create(
                points,
                grid_resolution=grid_resolution,
                z_min_threshold=z_min_threshold,
                z_max_threshold=z_max_threshold,
                skipCalibration=skipCalibration
            )
            frames.append(occupancy_grid)
            # printgrid(occupancy_grid, extent, f"VideoFrame_{i}")
        return frames, extent

    def videoPlayback(self, frames, extent, steps=None):  
        plt.ion()
        fig, ax = plt.subplots()
        img = plt.imshow(frames[0], cmap='binary', origin='upper', extent=extent)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('2D Occupancy Grid(Black is Occupied)')
        plt.tight_layout()
        fig.canvas.draw()
        for frame in frames[1:None if steps is None else steps]:
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
    


def testVid10sec10fps(record=False):
    kinect = Kinect()
    time.sleep(3)
    if record:
        video = kinect.record(duration=10, fps=10, playback=True)
        filename_10x10 = kinect.saveVideo(video, compress=False, filename="video10sec10fps.npy")
        compressed_filename_10x10 = kinect.saveVideo(video, compress=True, filename="video10sec10fps.npz")
        calibration_frame_10x10 = kinect.saveNPY(video[0], filename="calibration_frame_10x10.npy")
        calibration_frame = video[0]
    video = kinect.loadVideo("occupancypipe/videos/video10sec10fps.npy")
    calibration_frame = kinect.loadFrame("occupancypipe/frames/calibration_frame_10x10.npy", type='npy', view=False)
    kinect.calibrate(calibration_frame)
    frames_10x10, extent_10x10 = kinect.createVideo(video)
    kinect.videoPlayback(frames_10x10, extent_10x10)

def testVid5sec5fps(record=False):
    kinect = Kinect()
    time.sleep(3)
    if record:
        video = kinect.record(duration=5, fps=5, playback=True)
        filename_5x5 = kinect.saveVideo(video, compress=False, filename="video5sec5fps.npy")
        compressed_filename_5x5 = kinect.saveVideo(video, compress=True, filename="video5sec5fps.npz")
        calibration_frame_5x5 = kinect.saveNPY(video[0], filename="calibration_frame_5x5.npy")
        calibration_frame = video[0]
    video = kinect.loadVideo("occupancypipe/videos/video5sec5fps.npy")
    calibration_frame = kinect.loadFrame("occupancypipe/frames/calibration_frame_5x5.npy", type='npy', view=False)
    kinect.calibrate(calibration_frame)
    frames_5x5, extent_5x5 = kinect.createVideo(video)
    kinect.videoPlayback(frames_5x5, extent_5x5)
    


if __name__ == "__main__":
    testVid10sec10fps()