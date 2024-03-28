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

    angles1 = []
    angles2 = []
    xaxis1 = []
    xaxis2 = []
    threshold = 500
    
    for i, (p1, p2, p3, p4, p5, p6) in enumerate( zip(point1, point2, point3, point4, point5, point6)):
        dist = np.linalg.norm(np.array(p1) - np.array(p3))
        if dist > threshold:
            angles2.append(calculate_angle(p3, p4, p5, p6))
            xaxis2.append(i)
        else:
            angles1.append(calculate_angle(p3, p4, p5, p6))
            xaxis1.append(i)

    fig, axs = plt.subplots(1, 2)



    periods = round(10)  # Replace 10 with the desired number of periods

    angles1_diff = pd.Series(angles1).diff(periods=int(periods))
    angles2_diff = pd.Series(angles2).diff(periods=int(periods))

    axs[0].scatter(xaxis1, angles1_diff, color='black')
    axs[0].set_title('Zebrafish in SC')

    axs[1].scatter(xaxis2, angles2_diff, color='red')
    axs[1].set_title('Zebrafish in NSC')

    #max_val = max(np.max(angles1), np.max(angles2))
    #for ax in axs:
    #    ax.set_ylim(0, max_val)

    fig.subplots_adjust(wspace=0.5)

    plt.show()
    #results_df = pd.DataFrame({'Zebrafish in SC': angles1, 'Zebrafish in NSC': angles2})  TODO

    #results_df.to_csv('angles.csv', index=False)

if __name__ == "__main__":
    main()
