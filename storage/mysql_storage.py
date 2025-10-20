import mysql.connector
from mysql.connector import Error

def get_playlist_history(db_config, playlist_ids_array):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    placeholders = ",".join(["%s"] * len(playlist_ids_array))
    query = f"""
        SELECT
            r.report_date AS time,
            p.playlist_name AS metric,
            COUNT(d.video_id) AS value,
            r.playlist_id AS playlist_id
        FROM
            ytp_reports r
        JOIN
            ytp_report_details d ON r.report_id = d.report_id
        JOIN
            ytp_playlists p ON r.playlist_id = p.playlist_id
        WHERE
            r.playlist_id IN ({placeholders})
        GROUP BY
            r.report_date, p.playlist_name, r.playlist_id
        ORDER BY
            r.report_date ASC;
    """

    cursor.execute(query, playlist_ids_array)
    rows = cursor.fetchall()
    print(f"Pobrano {len(rows)} rekordów z bazy danych.")
    print(rows[:15])
    cursor.close()
    conn.close()
    return list(rows)