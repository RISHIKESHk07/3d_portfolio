import sqlite3
from datetime import datetime

class SentenceStorage:
    def __init__(self, db_path="sentences.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                timestamp DATETIME,
                processed BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()

    def add_sentence(self, text, processed=False):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sentences (text, timestamp, processed)
            VALUES (?, ?, ?)
        ''', (text, datetime.now(), processed))
        conn.commit()
        conn.close()

    def get_unprocessed(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, text FROM sentences WHERE processed = 0
        ''')
        data = cursor.fetchall()
        conn.close()
        return data

    def mark_processed(self, ids):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany('''
            UPDATE sentences SET processed = 1 WHERE id = ?
        ''', [(id,) for id in ids])
        conn.commit()
        conn.close(
        )
    def all_p_sentences(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT text FROM sentences WHERE processed = 1
        ''')
        data = cursor.fetchall()
        conn.close()
        return data
    
