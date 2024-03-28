import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(point1, point2, point3):
    
    vec1 = np.array(point1) - np.array(point2)
    vec2 = np.array(point3) - np.array(point2)

    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    angle_rad = np.arccos(dot_product / magnitude)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def main():
   
    df = pd.read_csv('predicted_body_coords.csv')

    x_fixed = df.iloc[2:, 1].astype(float)
    y_fixed = df.iloc[2:, 2].astype(float)

    x_head = df.iloc[2:, 4].astype(float)
    y_head = df.iloc[2:, 5].astype(float)

    x_body = df.iloc[2:, 7].astype(float)
    y_body = df.iloc[2:, 8].astype(float)

    
    fixed_point = list(zip(x_fixed, y_fixed))
    head = list(zip(x_head, y_head))
    body = list(zip(x_body, y_body))

    
    angles = [calculate_angle(fixed, head, body) for fixed, head, body in zip(fixed_point, head, body)]

   
    bins = np.linspace(0, 2*np.pi, 20, endpoint=True)
    hist, _ = np.histogram(angles, bins=bins)

    ax = plt.subplot(111, projection='polar')
    ax.bar(bins[:-1], hist, width=(bins[1]- bins[0]), bottom = 0.0 )
    plt.show()

if __name__ == "__main__":
    main()
