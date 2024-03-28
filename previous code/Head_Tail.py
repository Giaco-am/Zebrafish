import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import csv

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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

threshold = 500


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
    
    def analyze(self):
        
        point1 = list(zip(x_fixed, y_fixed))
        point2 = list(zip(x_hor, y_hor))
        point3 = list(zip(x_head, y_head))
        point4 = list(zip(x_body, y_body))

        angles1 = []
        angles2 = []
        frames = []

        for i, (p1, p2, p3, p4) in enumerate(zip(point1, point2, point3, point4)):
            dist = np.linalg.norm(np.array(p1) - np.array(p3))
            if dist > threshold:
                angles2.append(calculate_angle(p1, p2, p3, p4))
                angles1.append(np.nan)
                
            else:
                angles1.append(calculate_angle(p1, p2, p3, p4))
                angles2.append(np.nan)
            frames.append(i/158)


        max_len = max(len(angles1), len(angles2))
        angles1.extend([np.nan] * (max_len - len(angles1)))
        angles2.extend([np.nan] * (max_len - len(angles2)))
        frames.extend(range(len(angles1), max_len))


        n_bins = 40
        bins   = np.linspace(0, 2 * np.pi, n_bins, endpoint=True)

        hist1, _ = np.histogram(angles1, bins=bins)
        hist2, _ = np.histogram(angles2, bins=bins)

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

        axs[0].bar(bins[:-1], hist1, width=(bins[1] - bins[0]), bottom=0.0, color='black')
        axs[0].set_yticklabels([])
        axs[0].set_title('Zebrafish in SC')

        axs[1].bar(bins[:-1], hist2, width=(bins[1] - bins[0]), bottom=0.0, color='red')
        axs[1].set_yticklabels([])
        axs[1].set_title('Zebrafish in NSC')

        max_val = max(hist1.max(), hist2.max())
        for ax in axs:
            ax.set_ylim(0, max_val)

        fig.subplots_adjust(wspace=0.5)

        

        df_angles = pd.DataFrame({'Frames': frames,'Angles_SC': angles1, 'Angles_NSC': angles2})

        output_folder = 'HeadOrientationAnalysis_distance'
        create_directory(output_folder)
        df_angles.to_csv(os.path.join(output_folder,'head_orientation_angles.csv'), index=False)

        plt.savefig(os.path.join(output_folder,'head_orientation_analysis.png'))
        plt.show()

class TailMotionAnalysis:
    
    def __init__(self,x_fixed, y_fixed, x_head, y_head, x_body, y_body, x_tail, y_tail, x_tailend, y_tailend):
        
        self.point1 = list(zip(x_fixed,y_fixed))
        self.point3 = list(zip(x_head, y_head))
        self.point4 = list(zip(x_body, y_body))
        self.point5 = list(zip(x_tail, y_tail))
        self.point6 = list(zip(x_tailend, y_tailend))

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

        for i, (p1, p3, p4, p5, p6) in enumerate(zip(self.point1,self.point3, self.point4, self.point5, self.point6)):
            # Calculate distance between fish and box
            distance =  np.linalg.norm(np.array(p1) - np.array(p3))
            current_chamber = 'SC' if distance <= 500 else 'NSC'
            self.angles.append(calculate_angle(p3, p4, p5, p6))

            if self.prev_angle is not None:
                angle_diff = abs(self.angles[-1] - self.prev_angle)
                if angle_diff <= 5:
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
                self.colors.append('black')
            else:
                self.colors.append('red')

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
        self.output_folder = 'TailMotionAnalysis_distance'

        
        self.plot_chamber_durations()
        self.plot_tail_motion()


    def plot_tail_motion(self):
        
        plt.figure(figsize=(10, 6))
        plt.title('Tail motion')
        
        plt.scatter(self.seconds, self.angles_diff, c=self.colors, s=7)

        line_length = 5  
        for point in self.highlighted_points:
            plt.plot([self.seconds[point], self.seconds[point]], [self.angles_diff[point] - line_length, self.angles_diff[point] + line_length], 'b-', linewidth=0.1, alpha = 0.4)

        plt.xlabel('Time (seconds)')
        plt.ylabel(r'$\Delta\theta$ compared to previous frame')
        deg_format = plt.FuncFormatter(lambda x, _: '{:g}°'.format(x))
        plt.gca().yaxis.set_major_formatter(deg_format)

        handles = [Line2D([0], [0], color=c, marker='o', markersize=7, linestyle='') for c in ['black', 'red']]
        handles.append(Line2D([0], [0], color='blue', linestyle='-', linewidth=0.5))
        labels = ['Zebrafish in SC', 'Zebrafish in NSC', 'Freezing']
        plt.legend(handles, labels, loc='upper left')

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

        self.output_folder = 'TailOrientationAnalysis_distance'
        create_directory(self.output_folder)


        data = zip(sc_durations, sc_starts, nsc_durations, nsc_starts)
        csv_filename = os.path.join(self.output_folder, 'chamber_data.csv')

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SC Duration', 'SC Start', 'NSC Duration', 'NSC Start'])
            writer.writerows(data)

        plt.barh(0, sc_durations, color='black', left=sc_starts)
        plt.barh(1, nsc_durations, color='red', left=nsc_starts)
        plt.yticks([0, 1], ['SC', 'NSC'])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Chambers')
        plt.savefig(os.path.join(self.output_folder,'Chambers_bar_plot.png'))
        plt.show()

if __name__ == "__main__":

    head_analysis = HeadOrientationAnalysis()
    head_analysis.analyze()

    tail_analysis = TailMotionAnalysis(x_fixed, y_fixed, x_head, y_head, x_body, y_body, x_tail, y_tail, x_tailend, y_tailend)
    tail_analysis.analyze()
