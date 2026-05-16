import pandas as pd
import matplotlib.pyplot as plt


ROMP_CSV = r"C:\Users\Anuka\Desktop\7th Sem\EE7802 UGP\Github_Repo\HAR-FYP\Skin_Rehab\inputs\sprint_sequence.csv"
WHAM_CSV = r"C:\Users\Anuka\Desktop\7th Sem\EE7802 UGP\Github_Repo\HAR-FYP\Skin_Rehab\inputs\thetas.csv"
OUTPUT_IMAGE = "jitter_proof.png"


def main():
    romp_df = pd.read_csv(ROMP_CSV, header=None)
    wham_df = pd.read_csv(WHAM_CSV, header=None)

    frame_count = min(len(romp_df), len(wham_df), 373)
    frame_index = range(frame_count)

    romp_knee_x = romp_df.iloc[:frame_count, 12]
    wham_knee_x = wham_df.iloc[:frame_count, 12]

    plt.figure(figsize=(12, 6))
    plt.plot(
        frame_index,
        romp_knee_x,
        color="red",
        linewidth=1,
        alpha=0.7,
        label="ROMP (Frame-wise)",
    )
    plt.plot(
        frame_index,
        wham_knee_x,
        color="blue",
        linewidth=2.5,
        label="WHAM (Temporal)",
    )

    plt.title("Temporal Jitter Comparison: Frame-wise vs. Sequence Modeling")
    plt.xlabel("Frame Number")
    plt.ylabel("Axis-Angle Rotation (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.close()

    print(f"Saved plot to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
