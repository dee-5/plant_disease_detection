"""
Database module for Plant Disease Detection App.
Uses SQLite for zero-config, lightweight storage — ideal for edge devices.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scans.db')


def get_connection():
    """Get a database connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database — create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_filename TEXT NOT NULL,
            gradcam_filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            top3_json TEXT,
            explanation TEXT,
            is_healthy BOOLEAN DEFAULT 0,
            model_name TEXT DEFAULT 'mobilenet_v3'
        )
    ''')

    # Migration: add model_name column if table already exists without it
    cursor.execute("PRAGMA table_info(scans)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'model_name' not in columns:
        cursor.execute("ALTER TABLE scans ADD COLUMN model_name TEXT DEFAULT 'mobilenet_v3'")
        print("[DB] Migrated: added 'model_name' column to scans table")

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def save_scan(image_filename, gradcam_filename, predicted_class, confidence, top3, explanation, model_name='mobilenet_v3'):
    """Save a scan record to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    is_healthy = 1 if 'healthy' in predicted_class.lower() else 0
    cursor.execute('''
        INSERT INTO scans (image_filename, gradcam_filename, predicted_class, confidence, top3_json, explanation, is_healthy, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_filename, gradcam_filename, predicted_class, confidence, json.dumps(top3), explanation, is_healthy, model_name))
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id


def get_all_scans(limit=50):
    """Get all scans, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scans ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_scan(scan_id):
    """Get a single scan by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scans WHERE id = ?', (scan_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats():
    """Get summary statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as total FROM scans')
    total = cursor.fetchone()['total']

    cursor.execute('SELECT COUNT(*) as healthy FROM scans WHERE is_healthy = 1')
    healthy = cursor.fetchone()['healthy']

    cursor.execute('''
        SELECT predicted_class, COUNT(*) as count 
        FROM scans 
        GROUP BY predicted_class 
        ORDER BY count DESC 
        LIMIT 1
    ''')
    most_common_row = cursor.fetchone()
    most_common = most_common_row['predicted_class'] if most_common_row else 'N/A'

    conn.close()
    return {
        'total_scans': total,
        'healthy_count': healthy,
        'diseased_count': total - healthy,
        'health_ratio': round(healthy / total * 100, 1) if total > 0 else 0,
        'most_common': most_common
    }


def delete_scan(scan_id):
    """Delete a scan by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
    conn.commit()
    conn.close()


# Initialize on import
init_db()
