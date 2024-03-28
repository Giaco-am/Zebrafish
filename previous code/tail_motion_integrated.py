import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

def calculate_angle(point1, point2, point3, point4):

    vec1 = np.array(point1) - np.array(point2)
    vec2 = np.array(point3) - np.array(point4)

    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    angle_rad = np.arccos(dot_product / magnitude)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def main():

    df = pd.read_csv('pred_coords_35_mod.csv')

    x_fixed = df.iloc[2:, 1].astype(float)
    y_fixed = df.iloc[2:, 2].astype(float)

    x_hor = df.iloc[2:, 16].astype(float)
    y_hor = df.iloc[2:, 17].astype(float)

    x_head = df.iloc[2:, 4].astype(float)
    y_head = df.iloc[2:, 5].astype(float)

    x_body = df.iloc[2:, 7].astype(float)
    y_body = df.iloc[2:, 8].astype(float)

    x_tail = df.iloc[2:, 10].astype(float)
    y_tail = df.iloc[2:, 11].astype(float)

    x_tailend = df.iloc[2:, 13].astype(float)
    y_tailend = df.iloc[2:, 14].astype(float)

    point1 = list(zip(x_fixed,y_fixed))
    point2 = list(zip(x_hor, y_hor))
    point3 = list(zip(x_head, y_head))
    point4 = list(zip(x_body, y_body))
    point5 = list(zip(x_tail, y_tail))
    point6 = list(zip(x_tailend, y_tailend))
    
    angles = []
    xaxis = []
    colors = []
    
    threshold = 500

    for i, (p1, p2, p3, p4, p5, p6) in enumerate(zip(point1, point2, point3, point4, point5, point6)):
        dist = np.linalg.norm(np.array(p1) - np.array(p3))
        if dist > threshold:
            angles.append(calculate_angle(p3, p4, p5, p6))
            xaxis.append(i)
            colors.append('red')
            
        else:
            angles.append(calculate_angle(p3, p4, p5, p6))
            xaxis.append(i)
            colors.append('black')
            

    periods = 1   # To which previous frame you want the current frame to be compared

    angles_diff = pd.Series(angles).diff(periods=periods)

    
    plt.figure(figsize=(12, 8))
    plt.title('Tail motion')
    plt.scatter(xaxis, angles_diff, c=colors, s=7)

    plt.xlabel('Frames')
    plt.ylabel(r'$\Delta\theta$ compared to previous frame')
    deg_format = FuncFormatter(lambda x, _: '{:g}°'.format(x))
    plt.gca().yaxis.set_major_formatter(deg_format)
    
    handles = [Line2D([0], [0], color=c, marker='o', markersize=7, linestyle='') for c in ['black', 'red']]
    labels = ['Zebrafish in SC', 'Zebrafish in NSC']
    plt.legend(handles, labels, loc='upper left')


    plt.show()

if __name__ == "__main__":
    main()
