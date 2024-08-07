import numpy as np
from skimage.draw import polygon

class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []
            
    def __len__(self):
        return len(self.grs)

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        def _gr_text_to_no(l, offset=(0, 0)):
            """
            Transform a single point from a Cornell file line to a pair of ints.
            :param l: Line from Cornell grasp file (str)
            :param offset: Offset to apply to point positions
            :return: Point [x, y]
            """
            x, y = l.split()
            return [int(round(float(x))) - offset[0], int(round(float(y))) - offset[1]]
        
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([x, y]), w, h, -theta / 180.0 * np.pi).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        import matplotlib.pyplot as plt
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.height

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: height to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)
                
    @property
    def as_grasps(self):
        new_grasps = []
        for grasp_rectangle in self.grs:
            new_grasps.append(grasp_rectangle.as_grasp)
        return Grasps(new_grasps)
    
    @property
    def as_tlbra_grasps(self):
        new_tlbra_grasps = []
        for grasp_rectangle in self.grs:
            new_tlbra_grasps.append(grasp_rectangle.as_tlbra_grasp)
        return TLBRAGrasps(new_tlbra_grasps)

    def save_txt(self,txt_file):
        with open(txt_file, 'w') as file:
            for grasp_rectangle in self.grs:
                for point in grasp_rectangle.points: 
                    file.write(f'{point[0]} {point[1]}\n')
    
class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 0] - self.points[0, 0]
        dy = self.points[1, 1] - self.points[0, 1]
        return (np.arctan2(dy, -dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.width, self.height, self.angle)
    
    @property
    def tlbr(self):
        def rotate(points,angle,center):
            R = np.array(
                [
                    [np.cos(-angle), -1 * np.sin(-angle)],
                    [1 * np.sin(-angle), np.cos(-angle)],
                ]
            )
            c = np.array(center).reshape((1, 2))
            points_rotated = ((np.dot(R, (points - c).T)).T + c).astype(np.int)
            return points_rotated
        points_rotated = rotate(self.points,-self.angle,self.center)
        
        def find_tlbr(points):
            d = [points[i][0]**2 + points[i][1]**2 for i in range(len(points))]
            min_index = d.index(min(d))
            max_index = d.index(max(d))
            tl = self.points[min_index]
            br = self.points[max_index]
            return tl,br
        tl,br = find_tlbr(points_rotated)
        
        return tl,br
    
    @property
    def as_tlbra_grasp(self):
        tl,br = self.tlbr
        angle = self.angle
        return TLBRAGrasp(tl,br,angle)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)
    
    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dx = self.points[1, 0] - self.points[0, 0]
        dy = self.points[1, 1] - self.points[0, 1]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def height(self):
        """
        :return: Rectangle height (i.e. along the axis of the grasp)
        """
        dx = self.points[2, 0] - self.points[1, 0]
        dy = self.points[2, 1] - self.points[1, 1]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.width, self.height / 3, self.angle).as_gr.polygon_coords(shape)

    def iou(self, rectangle_Grasp, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        
        if abs((self.angle - rectangle_Grasp.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0.0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(rectangle_Grasp.points[:, 0], rectangle_Grasp.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0.0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0.0
        intersection = np.sum(canvas == 2)
        return intersection / union
    
    def get_all_metrics_info(self, rectangle_Grasps):
        all_metrics_info = []
        for rectangle_Grasp in rectangle_Grasps:
            metrics_info = {}
            metrics_info["self_angle"] = self.angle
            metrics_info["rectangle_Grasp_angle"] = rectangle_Grasp.angle
            metrics_info["angle_bias"] = (self.angle - rectangle_Grasp.angle + np.pi / 2) % np.pi - np.pi / 2
            
            rr1, cc1 = self.polygon_coords()
            rr2, cc2 = polygon(rectangle_Grasp.points[:, 0], rectangle_Grasp.points[:, 1])

            try:
                r_max = max(rr1.max(), rr2.max()) + 1
                c_max = max(cc1.max(), cc2.max()) + 1
                metrics_info["except_error"] = False
            except:
                metrics_info["except_error"] = True
                continue
            
            canvas = np.zeros((r_max, c_max))
            canvas[rr1, cc1] += 1
            canvas[rr2, cc2] += 1
            union = np.sum(canvas > 0)
            metrics_info["union_error"] = False
            if union == 0:
                metrics_info["union_error"] = True
            intersection = np.sum(canvas == 2)
            metrics_info["intersection"] = int(intersection)
            metrics_info["union"] = int(union)
            metrics_info["iou"] = intersection / union
            
            all_metrics_info.append(metrics_info)
        return all_metrics_info
    
    def get_metrics(self,rectangle_Grasps,iou_threshold=0.25, angle_threshold=np.pi / 6):
        metrics = {"first":0,"second":0,"both":0,"none":0}
        def metrics_angle(rectangle_Grasp,angle_threshold=np.pi / 6):
            if abs((self.angle - rectangle_Grasp.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
                return False
            else:
                return True
        def metrics_iou(rectangle_Grasp,iou_threshold=0.25):
            rr1, cc1 = self.polygon_coords()
            rr2, cc2 = polygon(rectangle_Grasp.points[:, 0], rectangle_Grasp.points[:, 1])
            try:
                r_max = max(rr1.max(), rr2.max()) + 1
                c_max = max(cc1.max(), cc2.max()) + 1
            except:
                return False
            canvas = np.zeros((r_max, c_max))
            canvas[rr1, cc1] += 1
            canvas[rr2, cc2] += 1
            union = np.sum(canvas > 0)
            if union == 0:
                iou = 0
            intersection = np.sum(canvas == 2)
            iou = intersection / union
            if iou<iou_threshold:
                return False
            else:
                return True
        for rectangle_Grasp in rectangle_Grasps:
            if metrics_angle(rectangle_Grasp,angle_threshold) and not metrics_iou(rectangle_Grasp,iou_threshold):
                metrics["first"] += 1
            elif not metrics_angle(rectangle_Grasp,angle_threshold) and metrics_iou(rectangle_Grasp,iou_threshold):
                metrics["second"] += 1
            elif metrics_angle(rectangle_Grasp,angle_threshold) and metrics_iou(rectangle_Grasp,iou_threshold):
                metrics["both"] += 1
            else:
                metrics["none"] += 1
        return metrics
    
    def max_iou(self, rectangle_Grasps):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param rectangle_Grasps: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        max_iou = 0
        for rectangle_Grasp in rectangle_Grasps:
            iou = self.iou(rectangle_Grasp)
            max_iou = max(max_iou, iou)
        return max_iou

    def get_all_iou(self,rectangle_Grasps):
        all_iou = []
        for rectangle_Grasp in rectangle_Grasps:
            all_iou.append(self.iou(rectangle_Grasp))
        return all_iou
    
    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [x, y] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), -1 * np.sin(-angle)],
                [1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 0], points[:, 1], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)
    
    def translate(self, x_shift, y_shift):
        """
        Translate the grasp rectangle by shifting points along the x and y axes.
        :param x_shift: Number of pixels to shift along the x axis.
        :param y_shift: Number of pixels to shift along the y axis.
        """
        self.points[:, 0] += x_shift
        self.points[:, 1] += y_shift

class Grasps:
    """
    Grasps represented by a center pixel, rotation angle and gripper width (height)
    """

    def __init__(self, grasps=None):
        if grasps:
            self.grasps = grasps
        else:
            self.grasps = []

    def __len__(self):
        return len(self.grasps)
    
    def __getitem__(self, item):
        return self.grasps[item]

    def __iter__(self):
        return self.grasps.__iter__()

    def __getattr__(self, attr):
        """
        Test if Grasp has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(Grasp, attr) and callable(getattr(Grasp, attr)):
            return lambda *args, **kwargs: list(map(lambda grasp: getattr(grasp, attr)(*args, **kwargs), self.grasps))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)
    
    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasps  from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: Grasps()
        """
        grasps = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                x = float(p0.split()[0])
                y = float(p0.split()[1])
                w = float(p0.split()[2])
                h = float(p0.split()[3])
                a = float(p0.split()[4])
                try:
                    grasps.append(Grasp([x,y],w,h,a))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grasps)
    
    @classmethod
    def load_from_list(cls,lst):
        grasps = []
        for i in range(len(lst)):
            x,y,w,h,a = lst[i]
            grasps.append(Grasp([x,y],w,h,a))
        return cls(grasps)
    
    @classmethod
    def load_from_grasp(cls,grasp):
        grasps = []
        x,y,w,h,a = grasp.get_data
        grasps.append(Grasp([x,y],w,h,a))
        return cls(grasps)
    
    @classmethod
    def load_from_jacquard_file(cls, fname):
        """
        Load grasps from a Jacquard dataset file.
        :param fname: Path to file.
        :return: Grasps()
        """
        grasps = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grasps.append(Grasp(np.array([x, y]), w, h, -theta / 180.0 * np.pi))
        return cls(grasps)
    
    def save_txt(self,txt_file):
        with open(txt_file, 'w') as file:
            for grasp in self.grasps:
                file.write(f'{grasp.center[0]} {grasp.center[1]} {grasp.width} {grasp.height} {grasp.angle}\n')
                
    @property
    def as_grs(self):
        new_grs = []
        for grasp in self.grasps:
            new_grs.append(grasp.as_gr)
        return GraspRectangles(new_grs)

    @property
    def as_tlbra_grasps(self):
        new_tlbra_grasps = []
        for grasp in self.grasps:
            new_tlbra_grasps.append(grasp.as_tlbra_grasp)
        return TLBRAGrasps(new_tlbra_grasps)

    @property
    def to_list(self):
        grasps_list = []
        for i in range(len(self.grasps)):
            grasp = self.grasps[i]
            grasps_list.append([grasp.center[0],grasp.center[1],grasp.width,grasp.height,grasp.angle])
        grasps_list = np.array(grasps_list).astype(float) # int32 to int
        return grasps_list.tolist()
    
    def append(self, grasp):
        """
        Add a grasp to this Grasps object
        :param grasp: Grasp
        """
        self.grasps.append(grasp)
    
    def get_max_w(self):
        widths = []
        for grasp in self.grasps:
            widths.append(grasp.width)
        return max(widths)
    
    def get_max_h(self):
        heights = []
        for grasp in self.grasps:
            heights.append(grasp.height)
        return sum(heights)/len(heights)
    
    def get_mean_x(self):
        xs = []
        for grasp in self.grasps:
            xs.append(grasp.center[0])
        return sum(xs)/len(xs)

    def get_mean_y(self):
        ys = []
        for grasp in self.grasps:
            ys.append(grasp.center[1])
        return sum(ys)/len(ys)
    
    def get_mean_w(self):
        widths = []
        for grasp in self.grasps:
            widths.append(grasp.width)
        return sum(widths)/len(widths)
    
    def get_mean_h(self):
        heights = []
        for grasp in self.grasps:
            heights.append(grasp.height)
        return sum(heights)/len(heights)
    
    def get_mean_a(self):
        angles = []
        for grasp in self.grasps:
            angles.append(grasp.angle)
        return sum(angles)/len(angles)
    
    def get_all_x(self):
        xs = []
        for grasp in self.grasps:
            xs.append(grasp.center[0])
        return xs

    def get_all_y(self):
        ys = []
        for grasp in self.grasps:
            ys.append(grasp.center[1])
        return ys
    
    def get_all_w(self):
        widths = []
        for grasp in self.grasps:
            widths.append(grasp.width)
        return widths
    
    def get_all_h(self):
        heights = []
        for grasp in self.grasps:
            heights.append(grasp.height)
        return heights
    
    def get_all_a(self):
        angles = []
        for grasp in self.grasps:
            angles.append(grasp.angle)
        return angles
    
    def get_numerical_bias(self,grasp):
        import math
        x_bias = []
        y_bias = []
        w_bias = []
        h_bias = []
        a_bias = []
        d_bias = []
        s_bias = []
        for self_grasp in self.grasps:
            x_bias.append(self_grasp.center[0]-grasp.center[0])
            y_bias.append(self_grasp.center[1]-grasp.center[1])
            w_bias.append(self_grasp.width-grasp.width)
            h_bias.append(self_grasp.height-grasp.height)
            a_bias.append(self_grasp.angle-grasp.angle)
            d_bias.append(math.sqrt((self_grasp.center[0]-grasp.center[0])**2+(self_grasp.center[0]-grasp.center[0])**2))
            s_bias.append(self_grasp.width*self_grasp.height-grasp.width*grasp.height)
        x_bias_abs = [abs(num) for num in x_bias]
        y_bias_abs = [abs(num) for num in y_bias]
        w_bias_abs = [abs(num) for num in w_bias]
        h_bias_abs = [abs(num) for num in h_bias]
        a_bias_abs = [abs(num) for num in a_bias]
        d_bias_abs = [abs(num) for num in d_bias]
        s_bias_abs = [abs(num) for num in s_bias]
        bias_dict = {"x_min_bias":min(x_bias_abs),
                     "x_max_bias":max(x_bias_abs),
                     "x_mean_bias":sum(x_bias_abs)/len(x_bias_abs),
                     "y_min_bias":min(y_bias_abs),
                     "y_max_bias":max(y_bias_abs),
                     "y_mean_bias":sum(y_bias_abs)/len(y_bias_abs),
                     "w_min_bias":min(w_bias_abs),
                     "w_max_bias":max(w_bias_abs),
                     "w_mean_bias":sum(w_bias_abs)/len(w_bias_abs),
                     "h_min_bias":min(h_bias_abs),
                     "h_max_bias":max(h_bias_abs),
                     "h_mean_bias":sum(h_bias_abs)/len(h_bias_abs),
                     "a_min_bias":min(a_bias_abs),
                     "a_max_bias":max(a_bias_abs),
                     "a_mean_bias":sum(a_bias_abs)/len(a_bias_abs),
                     "d_min_bias":min(d_bias_abs),
                     "d_max_bias":max(d_bias_abs),
                     "d_mean_bias":sum(d_bias_abs)/len(d_bias_abs),
                     "s_min_bias":min(s_bias_abs),
                     "s_max_bias":max(s_bias_abs),
                     "s_mean_bias":sum(s_bias_abs)/len(s_bias_abs)
        }
        return bias_dict
    

class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (height)
    """

    def __init__(self, center, width=30, height=60, angle=0):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.

    @property
    def get_data(self):
        return self.center[0],self.center[1],self.width,self.height,self.angle
    
    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.sin(self.angle)
        yo = np.cos(self.angle)

        x1 = self.center[0] - self.height / 2 * xo
        x2 = self.center[0] + self.height / 2 * xo
        y1 = self.center[1] - self.height / 2 * yo
        y2 = self.center[1] + self.height / 2 * yo

        return GraspRectangle(np.array(
            [
                [x1 - self.width / 2 * yo, y1 + self.width / 2 * xo],
                [x1 + self.width / 2 * yo, y1 - self.width / 2 * xo],
                [x2 + self.width / 2 * yo, y2 - self.width / 2 * xo],
                [x2 - self.width / 2 * yo, y2 + self.width / 2 * xo],
            ]
        ).astype(np.float))
    
    @property
    def as_tlbra_grasp(self):
        return self.as_gr.as_tlbra_grasp

    def get_all_metrics_info(self,Grasps):
        all_metrics_info = self.as_gr.get_all_metrics_info(Grasps.as_grs)
        return all_metrics_info
    
    def get_metrics(self,Grasps,iou_threshold=0.25, angle_threshold=np.pi / 6):
        metrics = self.as_gr.get_metrics(Grasps.as_grs,iou_threshold,angle_threshold)
        return metrics
    
    def max_iou(self, Grasps):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of Grasps
        :return: Maximum IoU with any of the GraspRectangles
        """
        max_iou = self.as_gr.max_iou(Grasps.as_grs)
        return max_iou
    
    def iou(self,Grasp):
        iou = self.as_gr.iou(Grasp.as_gr)
        return iou
    
    def get_all_iou(self,Grasps):
        all_iou = self.as_gr.get_all_iou(Grasps.as_grs)
        return all_iou
    
    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[0] * scale, self.center[1] * scale, -1 * self.angle * 180 / np.pi, self.height * scale,
            self.width * scale)
        
    @property
    def a_encoded(self):
        return TLBRAGrasp.encode_value(self.angle)

class TLBRAGrasps:
    def __init__(self, tlbra_grasps=None):
        if tlbra_grasps:
            self.tlbra_grasps = tlbra_grasps
        else:
            self.tlbra_grasps = []

    def __len__(self):
        return len(self.tlbra_grasps)
    
    def __getitem__(self, item):
        return self.tlbra_grasps[item]

    def __iter__(self):
        return self.tlbra_grasps.__iter__()

    def __getattr__(self, attr):
        """
        Test if Grasp has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(TLBRAGrasp, attr) and callable(getattr(TLBRAGrasp, attr)):
            return lambda *args, **kwargs: list(map(lambda tlbra_grasp: getattr(tlbra_grasp, attr)(*args, **kwargs), self.tlbra_grasps))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)
    
    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load tlbra_grasps  from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: TLBRAGrasps()
        """
        tlbra_grasps = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                tl_x = float(p0.split()[0])
                tl_y = float(p0.split()[1])
                br_x = float(p0.split()[2])
                br_y = float(p0.split()[3])
                a = float(p0.split()[4])
                try:
                    tlbra_grasps.append(TLBRAGrasp([tl_x,tl_y],[br_x,br_y],a))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(tlbra_grasps)
    
    def append(self, tlbra_grasp):
        self.tlbra_grasps.append(tlbra_grasp)

    def copy(self):
        new_tlbra_grasps = TLBRAGrasps()
        for tlbra_grasp in self.tlbra_grasps:
            new_tlbra_grasps.append(tlbra_grasp.copy())
        return new_tlbra_grasps

    def save_txt(self,txt_file):
        with open(txt_file, 'w') as file:
            for tlbra_grasp in self.tlbra_grasps:
                file.write(f'{tlbra_grasp.tl[0]} {tlbra_grasp.tl[1]} {tlbra_grasp.br[0]} {tlbra_grasp.br[1]} {tlbra_grasp.angle}\n')
    
    def save_txt_encoded(self,txt_file):
        with open(txt_file, 'w') as file:
            for tlbra_grasp in self.tlbra_grasps:
                file.write(f'{tlbra_grasp.tl_encoded} {tlbra_grasp.br_encoded} {tlbra_grasp.a_encoded}\n')

    @property
    def as_grs(self):
        new_grs = []
        for tlbra_grasp in self.tlbra_grasps:
            new_grs.append(tlbra_grasp.as_gr)
        return GraspRectangles(new_grs)

    @property
    def as_grasps(self):
        new_grasps = []
        for tlbra_grasp in self.tlbra_grasps:
            new_grasps.append(tlbra_grasp.as_grasp)
        return Grasps(new_grasps)
    
    def get_mean_tl(self):
        tlxs = []
        tlys = []
        for tlbra_grasp in self.tlbra_grasps:
            tlxs.append(tlbra_grasp.tl[0])
            tlys.append(tlbra_grasp.tl[1])
        return [sum(tlxs)/len(tlxs),sum(tlys)/len(tlys)]
    
    def get_mean_br(self):
        brxs = []
        brys = []
        for tlbra_grasp in self.tlbra_grasps:
            brxs.append(tlbra_grasp.tl[0])
            brys.append(tlbra_grasp.tl[1])
        return [sum(brxs)/len(brxs),sum(brys)/len(brys)]
    
    def get_mean_a(self):
        angles = []
        for tlbra_grasp in self.tlbra_grasps:
            angles.append(tlbra_grasp.angle)
        return sum(angles)/len(angles)
    
class TLBRAGrasp:
    def __init__(self, tl, br, angle=0):
        self.tl = tl
        self.br = br
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
    
    @property
    def as_gr(self):
        return self.tlbra2grasp_Rectangle(self.tl,self.br,self.angle)

    @property
    def as_grasp(self):
        return self.as_gr.as_grasp

    @property
    def get_data(self):
        return self.tl,self.br,self.angle
    
    @property
    def tl_encoded(self):
        return self.encode_point(self.tl)

    @property
    def br_encoded(self):
        return self.encode_point(self.br)
    
    @property
    def a_encoded(self):
        return self.encode_value(self.angle)
    
    @classmethod
    def encode_point(self, point, img_w=224, img_h=224, patch_w=32, patch_h=32):
        x = point[0]
        y = point[1]
        patch_size_x = img_w // patch_w
        patch_size_y = img_h // patch_h
        patch_num_x = min(max(0, x // patch_size_x), patch_w - 1)
        patch_num_y = min(max(0, y // patch_size_y), patch_h - 1)
        point_encoded = patch_num_y * patch_w + patch_num_x + 1
        return point_encoded  
        
    @classmethod
    def decode_point(self, point_encoded, img_w=224, img_h=224, patch_w=32, patch_h=32):
        patch_size_x = img_w // patch_w
        patch_size_y = img_h // patch_h
        patch_num_x = (point_encoded - 1) % patch_w
        patch_num_y = (point_encoded - 1) // patch_w
        x = patch_num_x * patch_size_x + patch_size_x // 2
        y = patch_num_y * patch_size_y + patch_size_y // 2
        point_decoded = [x, y]
        return point_decoded
    
    @classmethod
    def encode_value(self, value, bins=256, lower_limit=-np.pi/2, upper_limit=np.pi/2):
        range_width = upper_limit - lower_limit
        normalized_value = value - lower_limit
        scaled_value = normalized_value / range_width
        encoded_value = max(0, min(round(scaled_value * bins), 255))
        return encoded_value
    
    @classmethod
    def decode_value(self, encoded_value, lower_limit=-np.pi/2, upper_limit=np.pi/2, bins=256):
        range_width = upper_limit - lower_limit
        scaled_value = encoded_value / bins
        normalized_value = scaled_value * range_width
        decoded_value = normalized_value + lower_limit
        return decoded_value
    
    @classmethod
    def tlbra2grasp_Rectangle(self,tl,br,a):
        tl_x = tl[0]
        tl_y = tl[1]
        br_x = br[0]
        br_y = br[1]
        
        # for special situation
        if a == 0.0:
            a = 0.000000001
            
        # get the top-right point
        ta = np.tan(a)
        tr_x = (tl_y-br_y+ta*tl_x+br_x/ta)/(ta+1/ta)
        tr_y = -ta*(tr_x-tl_x)+tl_y
        tr = [tr_x,tr_y]
        
        # get the bottom-left point
        bl_x = tl_x + br_x - tr_x
        bl_y = tl_y + br_y - tr_y
        bl = [bl_x,bl_y]
        
        # get the grasp_Rectangle
        grasp_Rectangle = GraspRectangle(np.array([tl,tr,br,bl]))
        
        return grasp_Rectangle