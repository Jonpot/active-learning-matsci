# Install Packages
import static_frame as sf
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime


def main():
    # Arguments
    dataset = 'ct_scans' # 'vedo_shapes/random_250_v0' or 'ct_scans'
    drop_columns = False # False or iterable Ex. ('feature_npz',)
    bus_corrupted = False # If Bus frame names get corrupted, use ct_metadata.csv instead

    # Main
    data_dir = Path(f"datasets/{dataset}")

    if bus_corrupted:
        ct_metadata = sf.Frame.from_csv(data_dir / 'ct_metadata.csv', encoding='utf8', name='ct_metadata')
        ct_metadata = ct_metadata.drop[['__index0__']]
        ct_scans = list(ct_metadata['ct_scan'].values)
    else:
        bus = sf.Bus.from_zip_npz(data_dir / f"{data_dir.name}_dfs.zip")
        ct_scans = list(bus.keys())

    dfs_zip = data_dir / f"{data_dir.name}_dfs.zip"
    
    print('Remaking Bus:')
    dfs = []
    for ct_scan in (pbar := tqdm(ct_scans)):
        pbar.set_description("    Loading Frames")
        scan_dir = data_dir / ct_scan
        scan_df_csv = scan_dir / f"{ct_scan}_pores_df.csv"

        df = sf.Frame.from_csv(scan_df_csv, encoding='utf8', name=ct_scan)
        #df = df.rename(ct_scan)
        df = df.drop[['__index0__']]

        if drop_columns:
            df = df.drop[drop_columns]

        dfs.append(df)
        df.to_csv(scan_df_csv)

    print("    Saving Bus")
    bus = sf.Bus.from_frames(dfs, name = f"{data_dir.name}").sort_index()
    bus.to_zip_npz(dfs_zip)

    print("    Printing Bus")
    bus = sf.Bus.from_zip_npz(dfs_zip)
    print(bus)


if __name__ == "__main__":
    # TODO: Params loader
    start_time = time.time()

    main()

    duration = round(((time.time() - start_time))/60, 2)

    if duration >= 60:
        duration = f"{round(((time.time() - start_time))/3600, 2)} hours"
    else:
        duration = f"{round(((time.time() - start_time))/60, 2)} minutes"

    timestamp = datetime.now().strftime("%m-%d-%Y at %H:%M:%S")

    print(f"\nFinished in {duration} on {timestamp}")
    