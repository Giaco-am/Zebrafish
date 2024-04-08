import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import csv
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches


#TODO   find smarter way to convert frames to seconds in for loops, if no frames are dropped to likelihood constraints than results are best.




csv_file_path = r'coordinates/pred_coords_35000.csv'    
df = pd.read_csv(csv_file_path)
df = df.iloc[2:]

likelihood_threshold = 0.0
likelihood_cols      = [3, 6, 9, 12]
mask                 = df.iloc[:, likelihood_cols].astype(float).ge(likelihood_threshold).all(axis=1)
df_filtered          = df[mask]

num_rows_dropped = len(df) - len(df_filtered)
print(f"Number of rows dropped by likelihood threshold: {num_rows_dropped}")

x_fixed   = df_filtered.iloc[2:, 1] .astype(float)
y_fixed   = df_filtered.iloc[2:, 2] .astype(float)
x_hor     = df_filtered.iloc[2:, 1] .astype(float) + 400
y_hor     = df_filtered.iloc[2:, 2] .astype(float)
x_head    = df_filtered.iloc[2:, 4] .astype(float)
y_head    = df_filtered.iloc[2:, 5] .astype(float)
x_body    = df_filtered.iloc[2:, 7] .astype(float)
y_body    = df_filtered.iloc[2:, 8] .astype(float)
x_tail    = df_filtered.iloc[2:, 10].astype(float)
y_tail    = df_filtered.iloc[2:, 11].astype(float)
x_tailend = df_filtered.iloc[2:, 13].astype(float)
y_tailend = df_filtered.iloc[2:, 14].astype(float)



# SC BOX

x_box1, y_box1 = 380, 850  
x_box2, y_box2 = 580, 850  
x_box3, y_box3 = 580, 1150  
x_box4, y_box4 = 380, 1150 

df_filtered.loc[:, 'x_box1'] = x_box1
df_filtered.loc[:, 'y_box1'] = y_box1
df_filtered.loc[:, 'x_box2'] = x_box2
df_filtered.loc[:, 'y_box2'] = y_box2
df_filtered.loc[:, 'x_box3'] = x_box3
df_filtered.loc[:, 'y_box3'] = y_box3
df_filtered.loc[:, 'x_box4'] = x_box4
df_filtered.loc[:, 'y_box4'] = y_box4



def is_fish_in_box(point, box):

    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = box

    if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) < 0:
        return False

    if (x3 - x2) * (point[1] - y2) - (y3 - y2) * (point[0] - x2) < 0:
        return False

    if (x4 - x3) * (point[1] - y3) - (y4 - y3) * (point[0] - x3) < 0:
        return False

    if (x1 - x4) * (point[1] - y4) - (y1 - y4) * (point[0] - x4) < 0:
        return False

    # If the point passed all the checks, it is inside the square
    return True

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_angle(point1, point2, point3, point4):
    vec1 = np.array(point1) - np.array(point2)
    vec2 = np.array(point3) - np.array(point4)

    dot_product = np.dot(vec1, vec2)
    magnitude   = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    angle_rad   = np.arccos(dot_product / magnitude)
    angle_deg   = np.degrees(angle_rad)

    return angle_deg


class HeadOrientationAnalysis:

    def __init__(self, x_fixed, y_fixed, x_hor, y_hor, x_head, y_head, x_body, y_body, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4):
        
        self.point1 = list(zip(x_fixed, y_fixed))
        self.point2 = list(zip(x_hor, y_hor))
        self.point3 = list(zip(x_head, y_head))
        self.point4 = list(zip(x_body, y_body))
        self.box = [(x_box1, y_box1), (x_box2, y_box2), (x_box3, y_box3), (x_box4, y_box4)]


    def analyze(self):

        self.angles1 = []
        self.angles2 = []
        self.seconds = []

        for i, (p1, p2, p3, p4) in enumerate(zip(self.point1, self.point2, self.point3, self.point4)):
            if is_fish_in_box(p4, self.box):
                self.angles1.append(calculate_angle(p1, p2, p3, p4))
                self.angles2.append(np.nan)
            else:
                self.angles2.append(calculate_angle(p1, p2, p3, p4))
                self.angles1.append(np.nan)
            self.seconds.append(i/158)

        self.process_data()
        self.plot_data()
        self.save_data()


    def process_data(self):

        max_len = max(len(self.angles1), len(self.angles2))
        self.angles1.extend([np.nan] * (max_len - len(self.angles1)))
        self.angles2.extend([np.nan] * (max_len - len(self.angles2)))
        self.seconds.extend(range(len(self.angles1), max_len))

        self.n_bins = 40
        self.bins = np.linspace(0, 2 * np.pi, self.n_bins, endpoint=True)

        self.hist1, _ = np.histogram(self.angles1, bins=self.bins)
        self.hist2, _ = np.histogram(self.angles2, bins=self.bins)


    def plot_data(self):

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

        axs[0].bar(self.bins[:-1], self.hist1, width=(self.bins[1] - self.bins[0]), bottom=0.0, color='red')
        axs[0].set_yticklabels([])
        axs[0].set_title('Zebrafish in SC')

        axs[1].bar(self.bins[:-1], self.hist2, width=(self.bins[1] - self.bins[0]), bottom=0.0, color='black')
        axs[1].set_yticklabels([])
        axs[1].set_title('Zebrafish in NSC')

        max_val = max(self.hist1.max(), self.hist2.max())
        for ax in axs:
            ax.set_ylim(0, max_val)

        fig.subplots_adjust(wspace=0.5)


    def save_data(self):

        self.output_folder = 'HeadOrientationAnalysis_smooth'
        os.makedirs(self.output_folder, exist_ok=True)

        df_angles = pd.DataFrame({'Time (seconds)': self.seconds,'Angles_SC': self.angles1, 'Angles_NSC': self.angles2})
        df_angles.to_csv(os.path.join(self.output_folder,'head_orientation_angles.csv'), index=False)

        plt.savefig(os.path.join(self.output_folder,'head_orientation_analysis.png'))
        plt.show()




class TailMotionAnalysis:

    def __init__(self, x_head, y_head, x_body, y_body, x_tail, y_tail, x_tailend, y_tailend, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4):
        
        self.point3 = list(zip(x_head, y_head))
        self.point4 = list(zip(x_body, y_body))
        self.point5 = list(zip(x_tail, y_tail))
        self.point6 = list(zip(x_tailend, y_tailend))
        self.box = [(x_box1, y_box1), (x_box2, y_box2), (x_box3, y_box3), (x_box4, y_box4)]


    def analyze(self):
        
        self.angles = []
        self.seconds = []
        self.colors = []
        self.highlighted_points = []
        self.chamber_changes = []
        self.chamber_durations = []

        self.prev_chamber = None
        self.prev_time = None
        self.prev_angle = None
        self.stable_count = 0
        self.stable_sequence_start = None

        for i, (p3, p4, p5, p6) in enumerate(zip(self.point3, self.point4, self.point5, self.point6)):
            current_chamber = 'SC' if is_fish_in_box(p4, self.box) else 'NSC'
            self.angles.append(calculate_angle(p3, p4, p5, p6))

            if self.prev_angle is not None:
                angle_diff = abs(self.angles[-1] - self.prev_angle)
                if angle_diff <= 4:
                    if self.stable_sequence_start is None:
                        self.stable_sequence_start = i
                    self.stable_count += 1
                else:
                    if self.stable_sequence_start is not None:
                        if (i - self.stable_sequence_start) >= 3 * 158:
                            for j in range(self.stable_sequence_start, i):
                                self.highlighted_points.append(j)
                        self.stable_sequence_start = None
                    self.stable_count = 0
            else:
                self.stable_count = 0

            if current_chamber == 'SC':
                self.colors.append('red')
            else:
                self.colors.append('black')

            self.seconds.append(i/158)

            if self.prev_chamber is not None:
                if self.prev_chamber != current_chamber:
                    self.chamber_changes.append(i/158)
                    self.chamber_durations.append((i - self.prev_time)/158)
                    self.prev_time = i
                elif i == len(self.point3) - 1:
                    self.chamber_durations.append((i - self.prev_time + 1)/158)
            else:
                self.prev_time = i

            self.prev_chamber = current_chamber
            self.prev_angle = self.angles[-1]

        self.periods = 1
        self.angles_diff = pd.Series(self.angles).diff(periods=self.periods)
        self.output_folder = 'TailMotionAnalysis_smooth'
        os.makedirs(self.output_folder, exist_ok=True)

        #self.plot_tail_motion()
        self.plot_chamber_durations()
        self.save_data()
        #self.plot_polar_heatmap()
        self.plot_tail_motion_smooth()
    
  





    def plot_polar_heatmap(self):
        angles_diff = np.abs(self.angles_diff)
        hist, theta_edges, r_edges = np.histogram2d(angles_diff, self.seconds, bins=[np.linspace(0, np.pi, 40), np.linspace(0, max(self.seconds), 40)])
        hist_smooth = gaussian_filter(hist.T, sigma=2)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        theta, r = np.meshgrid(theta_edges, r_edges)
        im = ax.pcolormesh(theta, r, hist_smooth, cmap='RdGy')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('')
        
        ax.set_rlim(0, max(self.seconds))  
        ax.set_thetalim(0, np.pi)  

        
        ax.set_title('Polar Heatmap')

        #plt.savefig(os.path.join(self.output_folder, 'polar_heatmap.png'))
        plt.show()
        plt.close()

   

    
    
    def plot_tail_motion_smooth(self):
       


        sc_starts, nsc_starts, sc_durations, nsc_durations = self.plot_chamber_durations()

        plt.figure(figsize=(10, 6))
        plt.title('Tail motion')

        
        data_length = min(len(self.seconds), len(self.angles_diff), len(self.colors))
        self.seconds = self.seconds[:data_length]
        self.angles_diff = self.angles_diff[:data_length]
        self.colors = self.colors[:data_length]

        angles_diff_series = pd.Series(np.abs(self.angles_diff))

        #alpha = 0.1
        #angles_diff_smooth = angles_diff_series.ewm(alpha=alpha).mean()

        
        window_size = 80
        gaussian_filter = ndimage.gaussian_filter1d(np.ones(window_size), sigma=2)
        gaussian_filter /= np.sum(gaussian_filter)

    
        angles_diff_smooth = np.convolve(angles_diff_series, gaussian_filter, mode='same')
   
        for i in range(len(self.seconds)-1):
            plt.fill_between([self.seconds[i], self.seconds[i+1]], 
                            angles_diff_smooth[i], angles_diff_smooth[i+1], 
                            color=self.colors[i], alpha=0.5)

        
        for sc_start, nsc_start, sc_duration, nsc_duration in zip(sc_starts, nsc_starts, sc_durations, nsc_durations):
            plt.axvspan(sc_start, min(sc_start + sc_duration, self.seconds[-1]), color='red', alpha=0.3)
            plt.axvspan(nsc_start, min(nsc_start + nsc_duration, self.seconds[-1]), color='black', alpha=0.3)

        if nsc_starts:
            last_nsc_start = nsc_starts[-1]
            #last_patch_width = self.seconds[-1] - last_nsc_start
            plt.axvspan(last_nsc_start, self.seconds[-1], color='black', alpha=0.3)
        #TODO write the previous if statement also for sc_start

        line_length = 2
        for point in self.highlighted_points:
            plt.plot([self.seconds[point], self.seconds[point]], 
                    [angles_diff_smooth[point] - line_length, angles_diff_smooth[point] + line_length], 
                    'b-', linewidth=0.1, alpha=0.4)


        plt.xlabel('Time (seconds)')
        plt.ylabel(r'$\Delta\theta$ compared to previous frame')
        deg_format = plt.FuncFormatter(lambda x, _: '{:g}°'.format(x))
        plt.gca().yaxis.set_major_formatter(deg_format)
        handles = [Line2D([0], [0], color=c, marker='o', markersize=7, linestyle='') for c in ['red', 'black']]
        handles.append(Line2D([0], [0], color='blue', linestyle='-', linewidth=0.5))
        labels = ['Zebrafish in SC', 'Zebrafish in NSC', 'Freezing']
        plt.legend(handles, labels, loc='upper left')

        plt.savefig(os.path.join(self.output_folder,'tail_motion_analysis.png'))
        plt.show()
        plt.close()








    def plot_tail_motion(self):

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        fig.set_size_inches(10, 6)
        ax.set_title('Tail motion', va='bottom')

        ax.scatter(abs(self.angles_diff), self.seconds, c=self.colors, s=7)

        #line_length = 5
        #for point in self.highlighted_points:
        #    ax.plot([self.angles_diff[point], self.angles_diff[point]], [self.seconds[point], self.seconds[point]], 'b-', linewidth=0.1, alpha = 0.4)

        ax.set_thetalim(0, np.pi)

        ax.set_rlabel_position(-22.5)  
        ax.set_rticks([self.seconds[0], self.seconds[-1]])  
        ax.set_rlabel_position(0)  
        ax.grid(True)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        handles = [Line2D([0], [0], color=c, marker='o', markersize=7, linestyle='') for c in ['red', 'black']]
        handles.append(Line2D([0], [0], color='blue', linestyle='-', linewidth=0.5))
        labels = ['Zebrafish in SC', 'Zebrafish in NSC', 'Freezing']
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.savefig(os.path.join(self.output_folder,'tail_motion_analysis.png'))
        plt.show()
        plt.close()


    def plot_chamber_durations(self):

        chambers = ['SC'] * len(self.chamber_durations)
        chambers = [chamber if i%2 == 0 else 'NSC' for i, chamber in enumerate(chambers)]
        cumulative_durations = [0]

        filtered_chamber_durations = []
        filtered_chambers = []

        for i in range(len(self.chamber_durations)):
            if self.chamber_durations[i] >= 0.1:
                filtered_chamber_durations.append(self.chamber_durations[i])
                filtered_chambers.append(chambers[i])
                cumulative_durations.append(cumulative_durations[-1] + self.chamber_durations[i])

        plt.figure(figsize=(10, 4))
        plt.title('Chamber durations')
        sc_durations = [d for d, c in zip(filtered_chamber_durations, filtered_chambers) if c == 'SC']
        nsc_durations = [d for d, c in zip(filtered_chamber_durations, filtered_chambers) if c == 'NSC']
        sc_starts = [cumulative_durations[i] for i, c in enumerate(filtered_chambers) if c == 'SC']
        nsc_starts = [cumulative_durations[i] for i, c in enumerate(filtered_chambers) if c == 'NSC']

        data = zip(sc_durations, sc_starts, nsc_durations, nsc_starts)
        csv_filename = os.path.join(self.output_folder, 'chamber_data.csv')

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SC Start','SC Duration', 'NSC Start','NSC Duration'])
            writer.writerows(data)

        plt.barh(0, sc_durations, color='red', left=sc_starts)
        plt.barh(1, nsc_durations, color='black', left=nsc_starts)
        plt.yticks([0, 1], ['SC', 'NSC'])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Chambers')
        plt.savefig(os.path.join(self.output_folder,'Chambers_bar_plot.png'))
        plt.show()

        return sc_starts, nsc_starts, sc_durations, nsc_durations



    def save_data(self):
        
        df_angles_diff = pd.DataFrame({'Time in seconds': self.seconds, 'Delta_theta': self.angles_diff, 'Color': self.colors})
        df_angles_diff.to_csv(os.path.join(self.output_folder,'tail_motion_angles_diff.csv'), index=False)

    



if __name__ == "__main__":

    head_analysis = HeadOrientationAnalysis( x_fixed, y_fixed, x_hor, y_hor, x_head, y_head, x_body, y_body, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4)
    head_analysis.analyze()

    tail_analysis = TailMotionAnalysis(x_head, y_head, x_body, y_body, x_tail, y_tail, x_tailend, y_tailend, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4)
    tail_analysis.analyze()
