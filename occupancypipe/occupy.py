import os
os.environ["LIBFREENECT2_LOGGER_LEVEL"] = "error"
os.environ["LIBUSB_DEBUG"] = "0"
import open3d as o3d
from time import perf_counter
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod
import matplotlib.pyplot as plt
from freenect2 import Device, FrameType
import random
import time
from scipy import ndimage
from queue import Queue
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

    def transform1(self, points):
        ret = points.copy()
        ret[:, 0] = -points[:, 0]
        ret[:, 1] = -points[:, 1]
        ret[:, 2] = points[:, 2]
        return ret
    
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
                    reshape = points.reshape(-1, 3)
                    points = self.transform1(reshape)
                    pcd.points = o3d.utility.Vector3dVector(points)
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
    
    def calibrate(self, points, z_min_threshold=1.45, z_max_threshold=1.80):
        if z_min_threshold is not None and z_max_threshold is not None:
            filtered = points[(points[:, 2] >= z_min_threshold) & (points[:, 2] <= z_max_threshold)]
            if len(filtered) > 0:
                points = filtered
        self.x_min, self.x_max = points[:, 0].min(), points[:, 0].max()
        self.y_min, self.y_max = points[:, 1].min(), points[:, 1].max()
        self.calibrated = True
    
    def create(self, points, grid_resolution=0.02, z_min_threshold=1.45, z_max_threshold=1.80, skipCalibration=False, crop=20):
        table_points = points[(points[:, 2] >= z_min_threshold) & (points[:, 2] <= z_max_threshold)]
        # if len(table_points) == 0: return None, None
        if self.calibrated and not skipCalibration:
            x_min, x_max = self.x_min, self.x_max
            y_min, y_max = self.y_min, self.y_max
        else:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        grid_width = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
        grid_height = int(np.ceil((y_max - y_min) / grid_resolution)) + 1
        occupancy_grid = np.zeros((grid_height, grid_width))
        for point in table_points:
            x, y, _ = point
            if x < x_min or x > x_max or y < y_min or y > y_max:
                continue
            grid_x = int((x - x_min) / grid_resolution)
            grid_y = int((y - y_min) / grid_resolution)
            grid_x = max(0, min(grid_x, grid_width - 1))
            grid_y = max(0, min(grid_y, grid_height - 1))
            occupancy_grid[grid_y, grid_x] = 1
        occupancy_grid = occupancy_grid[crop:-crop, crop:-crop]
        x_min = x_min + (crop) * grid_resolution
        y_min = y_min + crop * grid_resolution
        x_max = x_max - (grid_width - crop) * grid_resolution
        y_max = y_max - (grid_height - crop) * grid_resolution
        return occupancy_grid, (x_min, x_max, y_min, y_max)

    def denoise(self, occupancy_grid, min_size=200 ):
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

    def createVideo(self, video, skipCalibration=False, grid_resolution=0.01, z_min_threshold=-1.9, z_max_threshold=-0.5, crop=20):
        frames = []
        for i, points in enumerate(video):
            occupancy_grid, extent = self.create(
                points,
                grid_resolution=grid_resolution,
                z_min_threshold=z_min_threshold,
                z_max_threshold=z_max_threshold,
                skipCalibration=skipCalibration,
                crop=crop
            )
            frames.append(occupancy_grid)
            # printgrid(occupancy_grid, extent, f"VideoFrame_{i}")
        print(frames[1].shape)
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

def run(record=False,convert=True, playback=True,  duration=10, fps=5, count=4):
    kinect = Kinect()
    # time.sleep(3)
    if record:
        video = kinect.record(duration=duration, fps=fps, playback=playback)
        kinect.saveVideo(video, compress=False, filename=f"video{duration}sec{fps}fps{count}.npy")
        kinect.saveVideo(video, compress=True, filename=f"video{duration}sec{fps}fps{count}.npz")
        kinect.saveNPY(video[0], filename=f"calibration_frame_{duration}x{fps}{count}.npy")
        calibration_frame = video[0]
    if convert:
        z_min_threshold=-2.8
        z_max_threshold=-1.5
        video = kinect.loadVideo(f"occupancypipe/videos/video{duration}sec{fps}fps{count}.npy")
        calibration_frame = kinect.loadFrame(f"occupancypipe/frames/calibration_frame_{duration}x{fps}{count}.npy", type='npy', view=False)
        kinect.calibrate(calibration_frame, z_min_threshold=z_min_threshold, z_max_threshold=z_max_threshold)

        frames_5x5, extent_5x5 = kinect.createVideo(
            video, 
            z_min_threshold=z_min_threshold, 
            z_max_threshold=z_max_threshold,
            crop=40)
        kinect.videoPlayback(frames_5x5, extent_5x5)
    


if __name__ == "__main__":
    # time.sleep(10)
    # run(record=False, convert=True, playback=True, duration=5, fps=5, count=4) # 4 is medium difficulty
    run(record=False, convert=True, playback=True, duration=5, fps=5, count=5) # 5 is easy
    # run(record=False, convert=True, playback=True, duration=5, fps=5, count=6) # 6 is hard
    