from datetime import datetime
import argparse
import os
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from storage import get_playlist_history

plt.rcParams['figure.dpi'] = 600


def auto_prophet_forecast(subset, display_name, CAP, TARGET_DATE):
    """
    Automatically configures Prophet based on playlist data characteristics.
    """

    subset = subset.sort_values('ds')
    avg_delta = subset['ds'].diff().dropna().median()

    # Detect frequency
    if pd.isna(avg_delta):
        freq = '2D'
    else:
        days = avg_delta.days
        if days <= 1:
            freq = '1D'
        elif days <= 3:
            freq = '2D'
        elif days <= 7:
            freq = '3D'
        else:
            freq = '7D'

    # Tune trend flexibility
    if freq == '1D':
        cps = 0.05
    elif freq == '2D':
        cps = 0.03
    elif freq == '3D':
        cps = 0.02
    else:
        cps = 0.015

    # Seasonalities
    yearly = len(subset) > 180
    weekly = freq in ['1D', '2D'] and len(subset) > 20

    # Growth type decision
    y_max = subset['y'].max()
    growth_type = 'linear' if y_max < CAP * 0.6 else 'logistic'

    print(f"[AutoProphet] Playlist: {display_name}")
    print(f"  -> freq={freq}, cps={cps}, growth={growth_type}, weekly={weekly}, yearly={yearly}")

    # Build Prophet model
    model = Prophet(
        growth=growth_type,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=False,
        changepoint_prior_scale=cps
    )

    df_train = subset[['ds', 'y']].copy()
    df_train['cap'] = CAP
    df_train['floor'] = 0

    if df_train['ds'].duplicated().any():
        df_train = df_train.sort_values('ds').groupby('ds', as_index=False).last()

    model.fit(df_train)

    # Forecast period
    last_date = subset['ds'].max()
    freq_offset = to_offset(freq)
    future_dates = pd.date_range(start=last_date + freq_offset, end=TARGET_DATE, freq=freq)
    periods = len(future_dates)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    future['cap'] = CAP
    future['floor'] = 0

    forecast = model.predict(future)

    # Smooth forecast for nicer visualization
    window = 5 if freq in ['1D', '2D'] else 3
    forecast['yhat_smooth'] = forecast['yhat'].rolling(window=window, center=True, min_periods=1).mean()
    forecast['yhat_lower_smooth'] = forecast['yhat_lower'].rolling(window=window, center=True, min_periods=1).mean()
    forecast['yhat_upper_smooth'] = forecast['yhat_upper'].rolling(window=window, center=True, min_periods=1).mean()

    for col in ['yhat_smooth', 'yhat_lower_smooth', 'yhat_upper_smooth']:
        forecast[col] = forecast[col].clip(lower=0, upper=CAP)

    return forecast, freq, cps, growth_type


def main():
    parser = argparse.ArgumentParser(description='Playlist Forecast Generator')
    parser.add_argument('-p', '--playlist_ids', nargs='+', help='Playlist IDs', required=True)
    args = parser.parse_args()
    playlist_ids = args.playlist_ids

    df = get_playlist_history(playlist_ids)
    if df is None or len(df) == 0:
        print("No data available for the given playlists.")
        return

    if not isinstance(df, pd.DataFrame):
        sample = df[0] if len(df) > 0 else ()
        if isinstance(sample, (list, tuple)) and len(sample) == 4:
            df = pd.DataFrame(df, columns=['time', 'metric', 'value', 'playlist_id'])
        else:
            df = pd.DataFrame(df, columns=['time', 'metric', 'value'])

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])

    TARGET_DATE = pd.to_datetime('2027-01-01')
    CAP = 5000

    # Playlist name mapping
    name_map = {}
    if 'playlist_id' in df.columns and 'metric' in df.columns:
        df['playlist_id'] = df['playlist_id'].astype(str)
        name_map = df.drop_duplicates('playlist_id').set_index('playlist_id')['metric'].to_dict()

    images_dir = os.path.join(os.getcwd(), "graphs")
    os.makedirs(images_dir, exist_ok=True)
    out_dir = images_dir
    combined_series = []

    requested_ids = [str(x) for x in playlist_ids]

    for pid in requested_ids:
        if 'playlist_id' in df.columns:
            subset = df[df['playlist_id'] == pid].copy()
        else:
            subset = df[df['metric'].astype(str) == pid].copy()

        if subset.empty:
            print(f"No historical data for playlist '{pid}'. Skipping.")
            continue

        display_name = name_map.get(pid, pid)

        subset = subset.rename(columns={'time': 'ds', 'value': 'y'})
        subset['y'] = subset['y'].clip(lower=0, upper=CAP)

        forecast, freq, cps, growth_type = auto_prophet_forecast(subset, display_name, CAP, TARGET_DATE)

        combined_series.append({
            'name': display_name,
            'ds': forecast['ds'].values,
            'forecast': forecast,
            'history': subset
        })

        # Individual chart
        plt.figure(figsize=(12, 6))
        plt.plot(subset['ds'], subset['y'], 'o-', label=f'{display_name} — historical', alpha=0.6)
        plt.plot(forecast['ds'], forecast['yhat_smooth'], label=f'{display_name} — forecast (smooth)')
        plt.fill_between(forecast['ds'], forecast['yhat_lower_smooth'], forecast['yhat_upper_smooth'], alpha=0.15)
        plt.title(f"Forecast of playlist size — {display_name}")
        plt.xlabel("Date")
        plt.ylabel("Number of tracks")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_name = "".join(c for c in str(display_name).strip().replace(' ', '_') if c.isalnum() or c in ('_', '-'))
        filename = f"{safe_name}_forecast_for_{TARGET_DATE.strftime('%Y-%m-%d')}_current_{datetime.now().strftime('%Y-%m-%d')}.png"
        filepath = os.path.join(out_dir, filename)
        plt.ylim(0, 1500)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved chart for playlist '{display_name}' ({growth_type}, {freq}) -> {filepath}")

    # Combined chart
    if len(combined_series) >= 2:
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('tab10')

        for i, s in enumerate(combined_series):
            color = cmap(i % 10)
            ds = pd.to_datetime(s['ds'])
            order = np.argsort(ds)
            forecast = s['forecast']
            hist = s['history']

            plt.plot(hist['ds'], hist['y'], 'o-', color=color, alpha=0.4, label=f"{s['name']} — history")
            plt.plot(ds.values[order], forecast['yhat_smooth'].values[order], linestyle='-', color=color,
                     label=f"{s['name']} — forecast")
            plt.fill_between(forecast['ds'], forecast['yhat_lower_smooth'], forecast['yhat_upper_smooth'],
                             color=color, alpha=0.15)

        plt.title("Playlist forecast comparison")
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
        plt.ylim(0, 1500)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved combined chart -> {filepath}")


if __name__ == "__main__":
    main()
