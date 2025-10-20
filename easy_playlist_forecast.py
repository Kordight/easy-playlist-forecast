from datetime import datetime
import argparse
import os
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

from storage import get_playlist_history

plt.rcParams['figure.dpi'] = 600

def main():
    parser = argparse.ArgumentParser(description='Easy Playlist Forecast')
    parser.add_argument('-p', '--playlist_ids', nargs='+', help='Playlist IDs', required=True)
    args = parser.parse_args()
    playlist_ids = args.playlist_ids

    # Fetch data from MySQL through the storage layer
    df = get_playlist_history(playlist_ids)
    if df is None or len(df) == 0:
        print("No data available for the given playlists.")
        return

    # Convert to DataFrame; handle both versions (with and without playlist_id column)
    if not isinstance(df, pd.DataFrame):
        sample = df[0] if len(df) > 0 else ()
        if isinstance(sample, (list, tuple)) and len(sample) == 4:
            df = pd.DataFrame(df, columns=['time', 'metric', 'value', 'playlist_id'])
        else:
            df = pd.DataFrame(df, columns=['time', 'metric', 'value'])

    # Debug info
    requested_ids = [str(x) for x in playlist_ids]
    print("requested_ids:", requested_ids)
    print("df.head():")
    print(df.head())

    if 'time' not in df.columns or 'value' not in df.columns:
        print("Expected columns ('time', 'value') not found in data.")
        return

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])

    TARGET_DATE = pd.to_datetime('2027-01-01')
    CAP = 5000

    # Prepare mapping from playlist_id to name if available
    name_map = {}
    if 'playlist_id' in df.columns and 'metric' in df.columns:
        df['playlist_id'] = df['playlist_id'].astype(str)
        name_map = df.drop_duplicates('playlist_id').set_index('playlist_id')['metric'].to_dict()

    out_dir = os.getcwd()
    images_dir = os.path.join(os.getcwd(), "graphs")
    os.makedirs(images_dir, exist_ok=True)
    out_dir = images_dir
    combined_series = []

    # Iterate through each requested playlist
    for pid in requested_ids:
        if 'playlist_id' in df.columns:
            subset = df[df['playlist_id'] == pid].copy()
        else:
            subset = df[df['metric'].astype(str) == pid].copy()

        if subset.empty:
            print(f"No historical data for playlist id '{pid}'. Skipping.")
            continue

        display_name = name_map.get(pid, pid)

        subset = subset.rename(columns={'time': 'ds', 'value': 'y'})
        subset = subset.sort_values('ds')
        subset['y'] = subset['y'].clip(lower=0, upper=CAP)

        # Prophet model setup
        model = Prophet(
            growth='logistic',
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )

        df_train = subset[['ds', 'y']].copy()
        df_train['cap'] = CAP
        df_train['floor'] = 0

        if df_train['ds'].duplicated().any():
            df_train = df_train.sort_values('ds').groupby('ds', as_index=False).last()

        model.fit(df_train)

        freq = pd.infer_freq(subset['ds'])
        if freq is None:
            freq = '2D'
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except Exception:
            offset = pd.tseries.frequencies.to_offset('1D')

        last_date = subset['ds'].max()
        if TARGET_DATE <= last_date:
            periods = 0
        else:
            future_dates = pd.date_range(start=last_date + offset, end=TARGET_DATE, freq=freq)
            periods = len(future_dates)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        future['cap'] = CAP
        future['floor'] = 0
        forecast = model.predict(future)

        forecast['yhat_orig_scale'] = forecast['yhat'].clip(lower=0, upper=CAP)
        forecast['yhat_lower_orig_scale'] = forecast['yhat_lower'].clip(lower=0, upper=CAP)
        forecast['yhat_upper_orig_scale'] = forecast['yhat_upper'].clip(lower=0, upper=CAP)

        forecast['yhat_int'] = forecast['yhat_orig_scale'].round().astype(int)
        forecast['yhat_lower_int'] = forecast['yhat_lower_orig_scale'].round().astype(int)
        forecast['yhat_upper_int'] = forecast['yhat_upper_orig_scale'].round().astype(int)

        # Store forecast for combined chart
        combined_series.append({
            'name': display_name,
            'ds': forecast['ds'].values,
            'forecast': forecast,
            'history': subset
        })

        # Individual plot
        plt.figure(figsize=(12, 6))
        plt.plot(subset['ds'], subset['y'], 'o-', label=f'{display_name} — historical data', alpha=0.6)
        plt.plot(forecast['ds'], forecast['yhat_orig_scale'], color='tab:blue',
                 label=f'{display_name} — forecast (float)')
        plt.plot(forecast['ds'], forecast['yhat_int'], color='tab:blue', linestyle='--',
                 label=f'{display_name} — forecast (int)')
        plt.fill_between(forecast['ds'], forecast['yhat_lower_orig_scale'], forecast['yhat_upper_orig_scale'],
                         color='tab:blue', alpha=0.15)

        plt.title(f"Forecast of playlist size — {display_name}")
        plt.xlabel("Date")
        plt.ylabel("Number of tracks")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_name = str(display_name).strip().replace(' ', '_')
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
        filename = f"{safe_name}_forecast_for_{TARGET_DATE.strftime('%Y-%m-%d')}_current_{datetime.now().strftime('%Y-%m-%d')}.png"
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved chart for playlist '{display_name}' (id {pid}) -> {filepath}")

    # Combined plot
    if len(combined_series) >= 2:
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('tab10')

        for i, s in enumerate(combined_series):
            color = cmap(i % 10)
            ds = pd.to_datetime(s['ds'])
            order = np.argsort(ds)

            forecast = s['forecast']
            hist = s['history']

            # Historical data
            plt.plot(hist['ds'], hist['y'], 'o-', color=color, alpha=0.4, label=f"{s['name']} — historical")

            # Forecast with confidence interval
            plt.plot(ds.values[order], forecast['yhat_orig_scale'].values[order],
                     linestyle='-', color=color, label=f"{s['name']} — forecast")
            plt.fill_between(forecast['ds'], forecast['yhat_lower_orig_scale'], forecast['yhat_upper_orig_scale'],
                             color=color, alpha=0.15)

        plt.title("Playlist forecast comparison with historical data")
        plt.xlabel("Date")
        plt.ylabel("Number of tracks")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        join_names = "_".join(
            "".join(c for c in s['name'] if c.isalnum() or c in ('_', '-')).replace(' ', '_')
            for s in combined_series
        )
        filename = f"combined_{join_names}_forecast_for_{TARGET_DATE.strftime('%Y-%m-%d')}_current_{datetime.now().strftime('%Y-%m-%d')}.png"
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved combined chart -> {filepath}")

if __name__ == "__main__":
    main()
