import sqlite3

def create_database():
    connection = sqlite3.connect("database/abusive_text.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT,
            severity TEXT,
            confidence REAL,
            action TEXT
        )
    """)

    connection.commit()
    connection.close()
def get_reports():
    connection = sqlite3.connect("database/abusive_text.db")
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, text, category, severity, confidence, action
        FROM reports
        ORDER BY id DESC
    """)

    reports = cursor.fetchall()

    connection.close()

    return reports
if __name__ == "__main__":
    create_database()
    print("Database created successfully!")