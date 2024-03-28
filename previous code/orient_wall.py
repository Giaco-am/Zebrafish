import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    #print(y_hor)
    x_head = df.iloc[2:, 4].astype(float)
    y_head = df.iloc[2:, 5].astype(float)

    x_body = df.iloc[2:, 7].astype(float)
    y_body = df.iloc[2:, 8].astype(float)
    
    

    point1 = list(zip(x_fixed,y_fixed))
    point2 = list(zip(x_hor, y_hor))
    point3 = list(zip(x_head, y_head))
    point4 = list(zip(x_body, y_body))

    angles = [calculate_angle(p1, p2, p3, p4) for p1, p2, p3, p4 in zip(point1, point2, point3, point4)]


    n_bins=40
    bins = np.linspace(0, 2*np.pi, n_bins, endpoint=True)
    hist, _ = np.histogram(angles, bins=bins)

    ax = plt.subplot(111, projection='polar')
    ax.bar(bins[:-1], hist, width=(bins[1]- bins[0]), bottom = 0.0 )
    ax.set_yticklabels([])

    
    plt.show()

if __name__ == "__main__":
    main()
