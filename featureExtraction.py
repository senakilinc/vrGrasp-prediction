import pandas as pd
import numpy as np
import os
from typing import List

def extract_grasp_aperture(df: pd.DataFrame, joint1: str = "R_ThumbTip", joint2: str = "R_IndexTip") -> pd.DataFrame:
    """
    Extract grasp aperture (distance between two joints) over time.

    Args:
        df (pd.DataFrame): Transform log dataframe (cropped to one config).
        joint1 (str): Name of the first joint (e.g., 'R_ThumbTip').
        joint2 (str): Name of the second joint (e.g., 'R_IndexTip').

    Returns:
        pd.DataFrame: DataFrame with 'Timestamp' and 'Aperture' columns.
    """
    joint1_df = df[df["Name"] == joint1][["Timestamp", "PosX", "PosY", "PosZ"]].reset_index(drop=True)
    joint2_df = df[df["Name"] == joint2][["Timestamp", "PosX", "PosY", "PosZ"]].reset_index(drop=True)

    # Align timestamps
    timestamps = joint1_df["Timestamp"].values
    positions1 = joint1_df[["PosX", "PosY", "PosZ"]].values
    positions2 = joint2_df[["PosX", "PosY", "PosZ"]].values

    # Euclidean distance between joint1 and joint2
    apertures = np.linalg.norm(positions1 - positions2, axis=1)

    return pd.DataFrame({
        "Timestamp": timestamps,
        "Aperture": apertures
    })

def load_and_extract_apertures(
    folder: str,
    joint1: str = "R_ThumbTip",
    joint2: str = "R_IndexTip"
) -> pd.DataFrame:
    """
    Load multiple transformed config files and extract grasp aperture for each.

    Args:
        folder (str): Folder containing cropped config transform CSVs.
        joint1 (str): First joint.
        joint2 (str): Second joint.

    Returns:
        pd.DataFrame: Combined DataFrame with columns ['Timestamp', 'Aperture', 'Object', 'ConfigFile']
    """
    all_data = []

    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue

        file_path = os.path.join(folder, fname)
        df = pd.read_csv(file_path)

        # Extract grasp aperture
        aperture_df = extract_grasp_aperture(df, joint1, joint2)
        aperture_df["Object"] = parse_object_name(fname)
        aperture_df["ConfigFile"] = fname

        all_data.append(aperture_df)

    return pd.concat(all_data, ignore_index=True)

def parse_object_name(filename: str) -> str:
    """
    Extract object name from filename, e.g., 'User0_TransformLog_BigCube_Grasp_config19.csv' → 'BigCube'
    """
    parts = filename.split("_")
    if len(parts) >= 4:
        return parts[2]  # Assuming structure: User0_TransformLog_BigCube_Grasp_config19.csv
    return "Unknown"

def extract_grasp_polygons_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts 3D edge vectors of 2 hand polygons (tip and intermediate) over time.
    Returns a DataFrame with one row per timestamp.

    Args:
        df (pd.DataFrame): Transform log with ['Timestamp', 'Name', 'PosX', 'PosY', 'PosZ']

    Returns:
        pd.DataFrame: Feature vectors per timestamp (30 features + Timestamp)
    """
    df["Name"] = df["Name"].str.strip()

    polygons = {
        "tip": [
            "R_ThumbTip", "R_IndexTip", "R_MiddleTip", "R_RingTip", "R_LittleTip"
        ],
        "intermediate": [
            "R_ThumbDistal", "R_IndexIntermediate", "R_MiddleIntermediate",
            "R_RingIntermediate", "R_LittleIntermediate"
        ]
    }

    grouped = df.groupby("Timestamp")
    features_per_frame = []

    for t, frame in grouped:
        frame = frame.set_index("Name")
        feature_row = {"Timestamp": t}
        complete = True

        for poly_name, joints in polygons.items():
            points = []
            for joint in joints:
                if joint not in frame.index:
                    complete = False
                    break
                pos = frame.loc[joint][["PosX", "PosY", "PosZ"]].values.astype(float)
                points.append(pos)

            if not complete:
                break

            points = np.array(points)
            points = np.vstack([points, points[0]])  # Close polygon

            edges = points[1:] - points[:-1]

            for i, edge in enumerate(edges):
                feature_row[f"{poly_name}_edge{i}_x"] = edge[0]
                feature_row[f"{poly_name}_edge{i}_y"] = edge[1]
                feature_row[f"{poly_name}_edge{i}_z"] = edge[2]

        if len(feature_row) > 1:
            features_per_frame.append(feature_row)

    return pd.DataFrame(features_per_frame)