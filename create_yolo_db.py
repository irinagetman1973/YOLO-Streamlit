import sqlite3

# Connect to the database
conn = sqlite3.connect('products.db')
cursor = conn.cursor()

# Sample data for YOLOv6, YOLOv7, YOLOv8, and YOLO-NAS
data = [
    ("yolov6", 0.85, 0.87, 0.86, 0.88),
    ("yolov7", 0.86, 0.88, 0.87, 0.89),
    ("yolov8", 0.87, 0.89, 0.88, 0.90),
    ("yolo-nas", 0.88, 0.90, 0.89, 0.91)
]

# Insert data into the yolo_metrics table
for entry in data:
    cursor.execute('''
    INSERT INTO yolo_metrics (model_version, precision, recall, f1_score, accuracy)
    VALUES (?, ?, ?, ?, ?)
    ''', entry)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Data inserted successfully!")


