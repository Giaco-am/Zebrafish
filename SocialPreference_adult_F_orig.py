import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import csv
from scipy import ndimage


#csv_file_path = r'coordinates\Adults F iso 1 spDLC_resnet50_social isolationApr3shuffle1_110000.csv'
coordinates_path = r'coordinates_orig'


for index,csv_file in enumerate(os.listdir(coordinates_path)):
    csv_file_path = os.path.join(coordinates_path, csv_file)

    subfolder_name = ' '.join(csv_file.split(' ')[2:4])


    df = pd.read_csv(csv_file_path)
    df = df.iloc[2:]

    likelihood_threshold = 0.0
    likelihood_cols = [3, 6, 9, 12]
    mask = df.iloc[:, likelihood_cols].astype(float).ge(likelihood_threshold).all(axis=1)
    df_filtered = df[mask]

    num_rows_dropped = len(df) - len(df_filtered)
    print(f"Number of rows dropped by likelihood threshold: {num_rows_dropped}")

    x_fixed = df_filtered.iloc[2:, 1].astype(float)
    y_fixed = df_filtered.iloc[2:, 2].astype(float)
    x_hor = df_filtered.iloc[2:, 1].astype(float) + 400
    y_hor = df_filtered.iloc[2:, 2].astype(float)
    x_head = df_filtered.iloc[2:, 4].astype(float)
    y_head = df_filtered.iloc[2:, 5].astype(float)
    x_body = df_filtered.iloc[2:, 7].astype(float)
    y_body = df_filtered.iloc[2:, 8].astype(float)
    x_tail = df_filtered.iloc[2:, 10].astype(float)
    y_tail = df_filtered.iloc[2:, 11].astype(float)
    x_tailend = df_filtered.iloc[2:, 13].astype(float)
    y_tailend = df_filtered.iloc[2:, 14].astype(float)

    # SC BOX
    x_box1, y_box1 = 360, 500
    x_box2, y_box2 = 490, 500
    x_box3, y_box3 = 490, 690
    x_box4, y_box4 = 360, 690

    df_filtered.loc[:, 'x_box1'] = x_box1
    df_filtered.loc[:, 'y_box1'] = y_box1
    df_filtered.loc[:, 'x_box2'] = x_box2
    df_filtered.loc[:, 'y_box2'] = y_box2
    df_filtered.loc[:, 'x_box3'] = x_box3
    df_filtered.loc[:, 'y_box3'] = y_box3
    df_filtered.loc[:, 'x_box4'] = x_box4
    df_filtered.loc[:, 'y_box4'] = y_box4

    # NSC BOX
    x_box5, y_box5 = 360, 30
    x_box6, y_box6 = 490, 30
    x_box7, y_box7 = 490, 230
    x_box8, y_box8 = 360, 230

    df_filtered.loc[:, 'x_box5'] = x_box5
    df_filtered.loc[:, 'y_box5'] = y_box5
    df_filtered.loc[:, 'x_box6'] = x_box6
    df_filtered.loc[:, 'y_box6'] = y_box6
    df_filtered.loc[:, 'x_box7'] = x_box7
    df_filtered.loc[:, 'y_box7'] = y_box7
    df_filtered.loc[:, 'x_box8'] = x_box8
    df_filtered.loc[:, 'y_box8'] = y_box8



    def is_fish_in_box_sc(point, box_sc):

        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = box_sc

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



    def is_fish_in_box_nsc(point, box_nsc):

        [(x5, y5), (x6, y6), (x7, y7), (x8, y8)] = box_nsc

        if (x6 - x5) * (point[1] - y5) - (y6 - y5) * (point[0] - x5) < 0:
            return False

        if (x7 - x6) * (point[1] - y6) - (y7 - y6) * (point[0] - x6) < 0:
            return False

        if (x8 - x7) * (point[1] - y7) - (y8 - y7) * (point[0] - x7) < 0:
            return False

        if (x5 - x8) * (point[1] - y8) - (y5 - y8) * (point[0] - x8) < 0:
            return False

        # If the point passed all the checks, it is inside the square
        return True




    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def calculate_angle(dot1, dot2, dot3, dot4):
        vec1 = np.array(dot1) - np.array(dot2)
        vec2 = np.array(dot3) - np.array(dot4)

        dot_product = np.dot(vec1, vec2)
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        angle_rad = np.arccos(dot_product / magnitude)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    class HeadOrientationAnalysis:
        def __init__(self, x_fixed, y_fixed, x_hor, y_hor, x_head, y_head, x_body, y_body, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4,
                    x_box5, y_box5, x_box6, y_box6, x_box7, y_box7, x_box8, y_box8):

            self.point1 = list(zip(x_fixed, y_fixed))
            self.point2 = list(zip(x_hor, y_hor))
            self.point3 = list(zip(x_head, y_head))
            self.point4 = list(zip(x_body, y_body))
            self.box_sc = [(x_box1, y_box1), (x_box2, y_box2), (x_box3, y_box3), (x_box4, y_box4)] 
            self.box_nsc=  [(x_box5, y_box5), (x_box6, y_box6), (x_box7, y_box7), (x_box8, y_box8)]
            self.angles1 = []
            self.angles2 = []
            self.angles3 = []

        def analyze(self):
            self.seconds = []

            for i, (p1, p2, p3, p4) in enumerate(zip(self.point1, self.point2, self.point3, self.point4)):
                if is_fish_in_box_sc(p4, self.box_sc):
                    self.angles1.append(calculate_angle(p1, p2, p3, p4))
                    self.angles2.append(np.nan)
                    self.angles3.append(np.nan)
                elif is_fish_in_box_nsc(p4, self.box_nsc):
                    self.angles1.append(np.nan)
                    self.angles2.append(calculate_angle(p1, p2, p3, p4))
                    self.angles3.append(np.nan)
                else:
                    self.angles1.append(np.nan)
                    self.angles2.append(np.nan)
                    self.angles3.append(calculate_angle(p1, p2, p3, p4))
                self.seconds.append(i/30)
            self.process_data()
            self.plot_and_save_data()

        def process_data(self):
            max_len = max(len(self.angles1), len(self.angles2), len(self.angles3))
            self.angles1.extend([np.nan] * (max_len - len(self.angles1)))
            self.angles2.extend([np.nan] * (max_len - len(self.angles2)))
            self.angles3.extend([np.nan] * (max_len - len(self.angles3)))
            self.seconds.extend(range(len(self.angles1), max_len))

            self.n_bins = 40
            self.bins = np.linspace(0, 2 * np.pi, self.n_bins, endpoint=True)

            self.hist1, _ = np.histogram(self.angles1, bins=self.bins)
            self.hist2, _ = np.histogram(self.angles2, bins=self.bins)
            self.hist3, _ = np.histogram(self.angles3, bins=self.bins)

        def plot_and_save_data(self):
    
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
            ax.bar(self.bins[:-1], self.hist1, width=(self.bins[1] - self.bins[0]), bottom=0.0, color='blue')
            ax.set_yticklabels([])
            ax.set_title(f'F_{subfolder_name}')
            max_val = self.hist1.max()
            ax.set_ylim(0, max_val)

            
            self.output_folder = 'Adults_orig'
            os.makedirs(self.output_folder, exist_ok=True)
            sub_folder = os.path.join(self.output_folder, 'SocialPref', f'F_{subfolder_name}')
            os.makedirs(sub_folder, exist_ok=True)
            df_angles = pd.DataFrame({'Time (seconds)': self.seconds, 'Angles_SC': self.angles1, 'Angles_NSC': self.angles2, 'Angles_neither': self.angles3})
            df_angles.to_csv(os.path.join(sub_folder, 'head_orientation_angles.csv'), index=False)

            plt.savefig(os.path.join(sub_folder, 'head_orientation_analysis.png'))
            plt.close()



    
    class TailMotionAnalysis:
        def __init__(self, x_head, y_head, x_body, y_body, x_tail, y_tail, x_tailend, y_tailend, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4, 
                    x_box5, y_box5, x_box6, y_box6, x_box7, y_box7, x_box8, y_box8):
            
            self.point3 = list(zip(x_head, y_head))
            self.point4 = list(zip(x_body, y_body))
            self.point5 = list(zip(x_tail, y_tail))
            self.point6 = list(zip(x_tailend, y_tailend))
            self.box_sc = [(x_box1, y_box1), (x_box2, y_box2), (x_box3, y_box3), (x_box4, y_box4)] 
            self.box_nsc=  [(x_box5, y_box5), (x_box6, y_box6), (x_box7, y_box7), (x_box8, y_box8)]

            self.angles1 = []
            self.angles2 = []
            self.angles3 = []
            self.colors = []

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
                current_chamber = 'SC' if is_fish_in_box_sc(p4, self.box_sc) else 'NSC' if is_fish_in_box_nsc(p4, self.box_nsc) else 'neither'
                self.angles.append(calculate_angle(p3, p4, p5, p6))

                if self.prev_angle is not None:
                    angle_diff = abs(self.angles[-1] - self.prev_angle)
                    if angle_diff <= 5:
                        if self.stable_sequence_start is None:
                            self.stable_sequence_start = i
                        self.stable_count += 1
                    else:
                        if self.stable_sequence_start is not None:
                            if (i - self.stable_sequence_start) >= 3 * 30:
                                for j in range(self.stable_sequence_start, i):
                                    self.highlighted_points.append(j)
                            self.stable_sequence_start = None
                        self.stable_count = 0
                else:
                    self.stable_count = 0

                if current_chamber == 'SC':
                    self.colors.append('blue')
                elif current_chamber == 'NSC':
                    self.colors.append('orange')
                else:
                    self.colors.append('gray')

                self.seconds.append(i/30)

                if self.prev_chamber is not None:
                    if self.prev_chamber != current_chamber:
                        self.chamber_changes.append(i/30)
                        self.chamber_durations.append((i - self.prev_time)/30)
                        self.prev_time = i
                    elif i == len(self.point3) - 1:
                        self.chamber_durations.append((i - self.prev_time + 1)/30)
                else:
                    self.prev_time = i

                self.prev_chamber = current_chamber
                self.prev_angle = self.angles[-1]

        

            self.periods = 1
            self.angles_diff = pd.Series(self.angles).diff(periods=self.periods)

            #self.process_data()
            self.plot_tail_motion_smooth()
            self.save_data()
            self.plot_chamber_durations()

        def process_data(self):
            max_len = max(len(self.angles1), len(self.angles2), len(self.angles3))
            self.angles1.extend([np.nan] * (max_len - len(self.angles1)))
            self.angles2.extend([np.nan] * (max_len - len(self.angles2)))
            self.angles3.extend([np.nan] * (max_len - len(self.angles3)))
            self.seconds.extend(range(len(self.angles1), max_len))
            self.colors.extend([' gray'] * (max_len - len(self.colors)))
            self.angles = [self.angles1[i] if not np.isnan(self.angles1[i]) else self.angles2[i] if not np.isnan(self.angles2[i]) else self.angles3[i] for i in range(max_len)]

            self.angle_differences = [np.nan]  
            for i in range(1, max_len):
                angle_diff = self.angles[i] - self.angles[i - 1]
                self.angle_differences.append(angle_diff)

    
                

        
        def plot_tail_motion_smooth(self):

            

            plt.figure(figsize=(16, 8))
            plt.title('Tail motion')

            

            angles_diff_series = pd.Series(np.abs(self.angles_diff))

            window_size = 80
            gaussian_filter = ndimage.gaussian_filter1d(np.ones(window_size), sigma=2)
            gaussian_filter /= np.sum(gaussian_filter)

            angles_diff_smooth = np.convolve(angles_diff_series, gaussian_filter, mode='same')

            for i in range(len(self.seconds)-1):
                    
                plt.fill_between([self.seconds[i], self.seconds[i+1]],
                            angles_diff_smooth[i], angles_diff_smooth[i+1],
                            color=self.colors[i], alpha=0.8)

        

            line_length = 2
            for point in self.highlighted_points:
                plt.plot([self.seconds[point], self.seconds[point]],
                        [angles_diff_smooth[point] - line_length, angles_diff_smooth[point] + line_length],
                        'y-', linewidth=0.1, alpha=0.3)
                
            plt.xlabel('Time (seconds)')
            plt.ylabel(r'$\Delta\theta$ compared to previous frame')
            deg_format = plt.FuncFormatter(lambda x, _: '{:g}°'.format(x))
            plt.gca().yaxis.set_major_formatter(deg_format)
            handles = [Line2D([0], [0], color=c, marker='o', markersize=7, linestyle='', linewidth=6) for c in ['blue', 'orange', 'gray']]
            handles.append(Line2D([0], [0], color='yellow', linestyle='-', linewidth=0.5))
            labels = ['Zebrafish in SC', 'Zebrafish in NSC','Zebrafish in Neither', 'Freezing']
            plt.legend(handles, labels, loc='upper left')
            self.output_folder = 'adults_orig'
            os.makedirs(self.output_folder, exist_ok=True)

            sub_folder = os.path.join(self.output_folder, 'SocialPref', f'F_{subfolder_name}')
            os.makedirs(sub_folder, exist_ok=True)

            plt.savefig(os.path.join(sub_folder,'tail_motion_analysis_smooth.png'))
            #plt.show()
            plt.close()





        def save_data(self):
            self.output_folder = 'adults_orig'
            os.makedirs(self.output_folder, exist_ok=True)

            sub_folder = os.path.join(self.output_folder, 'SocialPref', f'F_{subfolder_name}')
            os.makedirs(sub_folder, exist_ok=True)

            
            max_len = max(len(self.seconds), len(self.angles1), len(self.angles2), len(self.angles3))

            
            
            self.angles1 = self.angles1 + [np.nan] * (max_len - len(self.angles1))
            self.angles2 = self.angles2 + [np.nan] * (max_len - len(self.angles2))
            self.angles3 = self.angles3 + [np.nan] * (max_len - len(self.angles3))

        

            color_to_location = {'blue': 'SC', 'orange': 'NSC', 'gray': 'Neither'}
            locations = [color_to_location[color] for color in self.colors]

            df_angles = pd.DataFrame({
                'Time (seconds)': self.seconds, 
                'Angle difference':  np.abs(self.angles_diff),
                'Location': locations  
            })
            df_angles.to_csv(os.path.join(sub_folder,'tail_motion_angles.csv'), index=False)

            


        def plot_chamber_durations(self):
            total_time = len(self.point4) / 30
            segments = {'SC': [], 'NSC': [], 'Neither': []}

            current_box = None
            segment_start_time = 0
            previous_box = None
            
            
            for i, point in enumerate(self.point4):
                if is_fish_in_box_sc(point, self.box_sc):
                    new_box = 'SC'
                elif is_fish_in_box_nsc(point, self.box_nsc):
                    new_box = 'NSC'
                else:
                    new_box = 'Neither'

                if new_box != current_box:
                    if current_box is not None:
            
                        if (i / 30) - segment_start_time < 5 and previous_box is not None:
            
                            segments[previous_box][-1] = (segments[previous_box][-1][0], i / 30)
                        else:
            
                            segments[current_box].append((segment_start_time, i / 30))
                            previous_box = current_box
                        segment_start_time = i / 30
                    current_box = new_box

            
            if current_box is not None:
                segments[current_box].append((segment_start_time, total_time))

            self.output_folder = 'adults_orig'
            os.makedirs(self.output_folder, exist_ok=True)

            sub_folder = os.path.join(self.output_folder, 'SocialPref', f'F_{subfolder_name}')
            os.makedirs(sub_folder, exist_ok=True)
            self.csv_file= 'Chamber_time.csv'   
            

        
            self.csv_file= os.path.join(sub_folder, self.csv_file)


            with open(self.csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Chamber', 'Start Time', 'End Time'])
                for box, segment_list in segments.items():
                    for segment in segment_list:
                        start_time, end_time = segment
                        writer.writerow([box, start_time, end_time])


            
            for box, segment_list in segments.items():
                for segment in segment_list:
                    start_time, end_time = segment
                    duration = end_time - start_time
                    plt.barh(box, duration, left=start_time, color='blue' if box == 'SC' else 'orange' if box == 'NSC' else 'gray')

            plt.xlabel('Time (seconds)')
            plt.ylabel('Chambers')
            plt.title('Time spent in each chamber')

            self.output_folder = 'adults_orig'
            os.makedirs(self.output_folder, exist_ok=True)

            sub_folder = os.path.join(self.output_folder, 'SocialPref', f'F_{subfolder_name}')
            os.makedirs(sub_folder, exist_ok=True)

            plt.savefig(os.path.join(sub_folder,'chamber_time.png'))
            plt.close()

            #plt.show()

    
        
    if __name__ == "__main__":

        head_analysis = HeadOrientationAnalysis( x_fixed, y_fixed, x_hor, y_hor, x_head, y_head, x_body, y_body, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4,
                                            x_box5, y_box5, x_box6, y_box6, x_box7, y_box7, x_box8, y_box8)
        head_analysis.analyze()

        tail_analysis = TailMotionAnalysis( x_head, y_head, x_body, y_body, x_tail, y_tail, x_tailend, y_tailend, x_box1, y_box1, x_box2, y_box2, x_box3, y_box3, x_box4, y_box4,
                                           x_box5, y_box5, x_box6, y_box6, x_box7, y_box7, x_box8, y_box8)
        tail_analysis.analyze()
