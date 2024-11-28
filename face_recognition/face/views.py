from django.shortcuts import render
from django.http import HttpResponse
import getpass
import mysql.connector
from mysql.connector import errorcode
import hashlib
import cv2
import numpy as np
import pickle
from datetime import datetime, timedelta, date, time
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import pyttsx3
import os
from django.conf import settings

from django.http import HttpResponse
import configparser
from django.http import JsonResponse

def first(request):
        
    # Load configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')
    db_host = config['Database']['host']
    db_user = config['Database']['user']
    db_password = config['Database']['password']
    db_database = config['Database']['database']
    debug_mode = config.getboolean('General', 'debug')


    def create_connection():
        try:
            conn = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_database
            )
            return conn
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your username or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
            return None

    def create_tables():
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'employee'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT,
                name VARCHAR(100) NOT NULL,
                date DATE,
                check_in TIME,
                check_out TIME,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT,
                face_encoding BLOB,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        ''')

        # Insert admin user if not already present
        cursor.execute('''
            INSERT INTO employees (name, username, password, role) 
            SELECT 'Admin User', 'admin', %s, 'admin'
            FROM DUAL
            WHERE NOT EXISTS (
                SELECT 1 FROM employees WHERE username = 'admin'
            )
        ''', (hashlib.sha256('admin_password'.encode()).hexdigest(),))

        conn.commit()
        cursor.close()
        conn.close()

    create_tables()

    def capture_face_data(employee_id, num_samples=50):
        cascade_path = os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise IOError('Unable to load the face cascade classifier xml file')
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)  # Speed percent (can go over 100)
        engine.setProperty('volume', 0.9)  # Volume 0-1
        
        engine.say("Registered. Face the camera and get ready to capture your face data.")
        # engine.runAndWait()
        
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        faces_data = []
        i = 0


        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))  # Ensure consistent size
                if len(faces_data) < num_samples and i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, f"Samples captured: {len(faces_data)}/{num_samples}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if len(faces_data) >= num_samples:
                break
        
        video.release()
        cv2.destroyAllWindows()

        faces_data = np.asarray(faces_data)
        faces_data = faces_data.reshape(num_samples, -1)

        conn = create_connection()
        if conn is None:
            return

        cursor = conn.cursor()
        for face in faces_data:
            cursor.execute('''
                INSERT INTO face_data (employee_id, face_encoding)
                VALUES (%s, %s)
            ''', (employee_id, pickle.dumps(face)))

        conn.commit()
        cursor.close()
        conn.close()
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)  # Speed percent (can go over 100)
        engine.setProperty('volume', 0.9)  # Volume 0-1
        engine.say("Thank you. Face data captured and saved successfully.")
        engine.runAndWait()
        print("Face data captured and saved successfully.")


    def register_employee(name, username, password, role='employee'):
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute('SELECT COUNT(*) FROM employees WHERE username = %s', (username,))
        count = cursor.fetchone()[0]
        if count > 0:
            print("Username is already taken. Please try a different username.")
            cursor.close()
            conn.close()
            return
        
        try:
            cursor.execute('''
                INSERT INTO employees (name, username, password, role)
                VALUES (%s, %s, %s, %s)
            ''', (name, username, hashed_password, role))
            
            conn.commit()
            print("Employee registered successfully.")
            
            cursor.execute('SELECT id FROM employees WHERE username = %s', (username,))
            employee_id = cursor.fetchone()[0]
            
            print("Please look at the camera to capture your face data.")
            capture_face_data(employee_id)
            
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        
        cursor.close()
        conn.close()

    def train_knn_model():
        conn = create_connection()
        if conn is None:
            return None
        
        cursor = conn.cursor()
        cursor.execute('SELECT employee_id, face_encoding FROM face_data')
        records = cursor.fetchall()
        
        if not records:
            print("No face data found. Train the model after registering some employees.")
            cursor.close()
            conn.close()
            return None
        
        face_encodings = []
        labels = []
        
        for record in records:
            employee_id, face_encoding = record
            face_encodings.append(pickle.loads(face_encoding))
            labels.append(employee_id)
        
        knn_clf = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
        knn_clf.fit(face_encodings, labels)
        
        cursor.close()
        conn.close()
        return knn_clf

    def recognize_and_check(knn_clf, check_type='in'):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            raise IOError('Unable to load the face cascade classifier xml file')
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w, :]
                face = cv2.resize(face, (50, 50))  # Ensure consistent
                face_encoding = face.flatten().reshape(1, -1)
                
                employee_id = knn_clf.predict(face_encoding)[0]
                employee_id = int(employee_id)  # Convert numpy.int32 to int
                
                if check_type == 'in':
                    check_in(employee_id)
                else:
                    check_out(employee_id)
                
                cap.release()
                cv2.destroyAllWindows()
                return
        
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        


    def check_in(employee_id):
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        cursor.execute('SELECT name FROM employees WHERE id = %s', (employee_id,))
        name = cursor.fetchone()[0]
        
        date = datetime.now().strftime('%Y-%m-%d')
        check_in_time = datetime.now().strftime('%H:%M:%S')
        
        cursor.execute('''
            INSERT INTO attendance (employee_id, name, date, check_in) 
            VALUES (%s, %s, %s, %s)
        ''', (employee_id, name, date, check_in_time))
        
        conn.commit()
        cursor.close()
        conn.close()
        


        engine = pyttsx3.init()
        engine.setProperty('rate', 130)  # Speed percent (can go over 100)
        engine.setProperty('volume', 0.9)  # Volume 0-1
        
        engine.say("You have successfully checked in. Have a great day.")
        engine.runAndWait()
        

    def check_out(employee_id):
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        date = datetime.now().strftime('%Y-%m-%d')
        check_out_time = datetime.now().strftime('%H:%M:%S')
        
        cursor.execute('''
            UPDATE attendance SET check_out = %s 
            WHERE employee_id = %s AND date = %s
        ''', (check_out_time, employee_id, date))
        
        conn.commit()
        cursor.close()
        conn.close()
        

        engine = pyttsx3.init()
        engine.setProperty('rate', 130)  # Speed percent (can go over 100)
        engine.setProperty('volume', 0.9)  # Volume 0-1
        
        engine.say("You have successfully checked out. Thank you for your day at the company.")
        engine.runAndWait()

    def generate_report(employee_id, report_type='daily'):
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        if report_type == 'daily':
            date = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT date, check_in, check_out FROM attendance WHERE employee_id = %s AND date = %s
            ''', (employee_id, date))
            records = cursor.fetchall()
            
            if not records:
                print(f"No attendance records found for employee ID {employee_id} for the {report_type} report.")
            else:
                print(f"\n{report_type.capitalize()} Attendance Report for Employee ID {employee_id}:\n")
                for record in records:
                    print(f"Date: {record[0]}, Check-In: {record[1]}, Check-Out: {record[2]}")
        
        
        cursor.close()
        conn.close()

    def convert_timedelta_to_time(timedelta_obj):
        if timedelta_obj is None:
            return None
        total_seconds = int(timedelta_obj.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return time(hours, minutes, seconds)


    def generate_report(employee_id, report_type):
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        if report_type == 'daily':
            today = datetime.now().strftime('%Y-%m-%d')
            query = '''
                SELECT e.name, a.date, a.check_in, a.check_out FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.employee_id = %s AND a.date = %s
            '''
            cursor.execute(query, (employee_id, today))
        
        elif report_type == 'weekly':
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            start_of_week_str = start_of_week.strftime('%Y-%m-%d')
            end_of_week_str = end_of_week.strftime('%Y-%m-%d')
            
            query = '''
                SELECT e.name, a.date, a.check_in, a.check_out FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.employee_id = %s AND a.date BETWEEN %s AND %s
            '''
            cursor.execute(query, (employee_id, start_of_week_str, end_of_week_str))
        
        elif report_type == 'monthly':
            today = datetime.now()
            start_of_month = today.replace(day=1)
            end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            start_of_month_str = start_of_month.strftime('%Y-%m-%d')
            end_of_month_str = end_of_month.strftime('%Y-%m-%d')
            
            query = '''
                SELECT e.name, a.date, a.check_in, a.check_out FROM attendance a
                JOIN employees e ON a.employee_id = e.id
                WHERE a.employee_id = %s AND a.date BETWEEN %s AND %s
            '''
            cursor.execute(query, (employee_id, start_of_month_str, end_of_month_str))
        
        records = cursor.fetchall()
        
        if not records:
            print("No attendance records found.")
            cursor.close()
            conn.close()
            return
        
        

        # Convert timedelta to time for Check-In and Check-Out
        records = [(name, date, convert_timedelta_to_time(check_in), convert_timedelta_to_time(check_out)) for name, date, check_in, check_out in records]

        df = pd.DataFrame(records, columns=['Name', 'Date', 'Check-In', 'Check-Out'])
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert Check-In and Check-Out to datetime, handling errors
        df['Check-In'] = pd.to_datetime(df['Check-In'], format='%H:%M:%S', errors='coerce')
        df['Check-Out'] = pd.to_datetime(df['Check-Out'], format='%H:%M:%S', errors='coerce')
        
    
        
        # Calculate Duration, fill NaN durations with 0 (in case of missing Check-Out)
        df['Duration'] = (df['Check-Out'] - df['Check-In']).dt.total_seconds() / 3600
        df['Duration'] = df['Duration'].fillna(0)
        
        print(df)
        
        # Optionally generate graphs for weekly and monthly reports
        if report_type in ['weekly', 'monthly']:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Date', y='Duration', data=df, errorbar=None)
            plt.title(f'Attendance Duration for {report_type.capitalize()} Report')
            plt.xlabel('Date')
            plt.ylabel('Duration (hours)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
        
        cursor.close()
        conn.close()

    def generate_attendance_graph(df, report_type):
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate the duration between check-in and check-out
        df['Check-In'] = pd.to_datetime(df['Check-In'])
        df['Check-Out'] = pd.to_datetime(df['Check-Out'])
        df['Duration'] = (df['Check-Out'] - df['Check-In']).dt.total_seconds() / 3600  # Convert to hours
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Date', y='Duration', data=df, marker='o')
        plt.title(f'{report_type.capitalize()} Attendance Duration')
        plt.xlabel('Date')
        plt.ylabel('Duration (hours)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def generate_overall_attendance_graph():
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT e.name, a.date, a.check_in, a.check_out FROM attendance a
            JOIN employees e ON a.employee_id = e.id
        ''')
        records = cursor.fetchall()
        
        if not records:
            print("No attendance records found.")
            cursor.close()
            conn.close()
            return
        def convert_timedelta_to_time(timedelta_obj):
            if timedelta_obj is None:
                return None
            total_seconds = int(timedelta_obj.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return time(hours, minutes, seconds)

        # Convert timedelta to time for Check-In and Check-Out
        records = [(name, date, convert_timedelta_to_time(check_in), convert_timedelta_to_time(check_out)) for name, date, check_in, check_out in records]

        df = pd.DataFrame(records, columns=['Name', 'Date', 'Check-In', 'Check-Out'])
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert Check-In and Check-Out to datetime, handling errors
        df['Check-In'] = pd.to_datetime(df['Check-In'], format='%H:%M:%S', errors='coerce')
        df['Check-Out'] = pd.to_datetime(df['Check-Out'], format='%H:%M:%S', errors='coerce')
        
        
        
        # Calculate Duration, fill NaN durations with 0 (in case of missing Check-Out)
        df['Duration'] = (df['Check-Out'] - df['Check-In']).dt.total_seconds() / 3600
        df['Duration'] = df['Duration'].fillna(0)
        
        
        
        # Aggregate the total duration per employee
        df_total_duration = df.groupby('Name', as_index=False)['Duration'].sum()
        
    
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Name', y='Duration', data=df_total_duration, errorbar=None)
        plt.title('Total Work Duration for Each Employee')
        plt.xlabel('Employee Name')
        plt.ylabel('Duration (hours)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
        cursor.close()
        conn.close()    
        

    def view_all_employees():
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM employees')
        records = cursor.fetchall()
        
        for record in records:
            print(record)
        
        cursor.close()
        conn.close()

    def delete_employee(employee_id):
        conn = create_connection()
        if conn is None:
            return
        
        cursor = conn.cursor()
        
        try:
            # Delete associated attendance records
            cursor.execute('DELETE FROM attendance WHERE employee_id = %s', (employee_id,))
            
            # Delete associated face data records
            cursor.execute('DELETE FROM face_data WHERE employee_id = %s', (employee_id,))
            
            # Delete the employee
            cursor.execute('DELETE FROM employees WHERE id = %s', (employee_id,))
            
            conn.commit()
            print("Employee deleted successfully.")
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            conn.rollback()  # Rollback the transaction in case of error
        
        cursor.close()
        conn.close()


    def already_checked_in(employee_id):
        conn = create_connection()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE employee_id = %s AND date = %s AND check_in IS NOT NULL AND check_out IS NULL
        ''', (employee_id, date))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0

    def already_checked_out(employee_id):
        conn = create_connection()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE employee_id = %s AND date = %s AND check_out IS NOT NULL
        ''', (employee_id, date))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0

    def main():
        create_tables()
        
        while True:
            print("\n1. Register")
            print("2. Login")
            print("3. Exit")
            
            choice = input("Enter your choice: ")
            
            if choice == '1':
                name = input("Enter name: ")
                while True:
                    username = input("Enter username: ")
                    conn = create_connection()
                    if conn is None:
                        continue
                    
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM employees WHERE username = %s', (username,))
                    count = cursor.fetchone()[0]
                    cursor.close()
                    conn.close()
                    
                    if count > 0:
                        print("Username is already taken. Please try a different username.")
                    else:
                        break

                password = getpass.getpass("Enter password: ")
                
                # Selecting role
                print("Select role:")
                print("1. Admin")
                print("2. Employee")
                role_choice = input("Enter role choice: ")
                if role_choice == '1':
                    role = 'admin'
                elif role_choice == '2':
                    role = 'employee'
                else:
                    print("Invalid role choice. Defaulting to 'employee'.")
                    role = 'employee'
                
                register_employee(name, username, password, role)
            
            elif choice == '2':
                username = input("Enter username: ")
                password = getpass.getpass("Enter password: ")
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                
                conn = create_connection()
                if conn is None:
                    continue
                
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, role FROM employees WHERE username = %s AND password = %s
                ''', (username, hashed_password))
                result = cursor.fetchone()
                
                if result:
                    employee_id, role = result
                    print(f"Welcome, {role} {username}.")
                    
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 140)  # Speed percent (can go over 100)
                    engine.setProperty('volume', 0.9)  # Volume 0-1
                    engine.say("You have logged in. Welcome.")
                    print("Hello")
                    # engine.runAndWait()
                    print("BYE")
                    knn_clf = train_knn_model()
                    
                    if knn_clf is None:
                        print("Please register employees and capture face data first.")
                        continue
                    
                    if role == 'admin':
                        while True:
                            print("\nAdmin Menu:")
                            print("1. Generate Report")
                            print("2. View All Employees")
                            print("3. Delete Employee")
                            print("4. Check In")
                            print("5. Check Out")
                            print("6. Generate Overall Attendance graph")
                            print("7. Logout")
                            
                            admin_choice = input("Enter your choice: ")
                            
                            if admin_choice == '1':
                                employee_id_report = input("Enter employee ID for report: ").strip()
                                print("\nSelect report type:")
                                print("1. Daily")
                                print("2. Weekly")
                                print("3. Monthly")
                                report_choice = input("Enter report type choice: ")
                                if report_choice == '1':
                                    report_type = 'daily'
                                elif report_choice == '2':
                                    report_type = 'weekly'
                                elif report_choice == '3':
                                    report_type = 'monthly'
                                else:
                                    print("Invalid report type choice. Defaulting to 'daily'.")
                                    report_type = 'daily'
                                generate_report(employee_id_report, report_type)
                            elif admin_choice == '2':
                                view_all_employees()
                            elif admin_choice == '3':
                                employee_id_delete = input("Enter employee ID to delete: ")
                                delete_employee(employee_id_delete)
                            elif admin_choice == '4':
                                if already_checked_in(employee_id):
                                    print("You have already checked in today.")
                                else:
                                    recognize_and_check(knn_clf, check_type='in')
                                    print("You have successfully checked in.")
                                break  # Automatically log out after check-in
                            elif admin_choice == '5':
                                if not already_checked_in(employee_id):
                                    print("You need to check in before checking out.")
                                elif already_checked_out(employee_id):
                                    print("You have already checked out today.")
                                else:
                                    recognize_and_check(knn_clf, check_type='out')
                                    print("You have successfully checked out.")
                                break  # Automatically log out after check-out
                            elif admin_choice == '6':
                                generate_overall_attendance_graph()
                            elif admin_choice == '7':
                                print("You have been logged out.")
                                break
                            else:
                                print("Invalid choice. Please try again.")
                    
                    elif role == 'employee':
                        while True:
                            print("\nEmployee Menu:")
                            print("1. Check In")
                            print("2. Check Out")
                            print("3. Logout")
                            
                            
                            employee_choice = input("Enter your choice: ")
                            
                            if employee_choice == '1':
                                if already_checked_in(employee_id):
                                    print("You have already checked in today.")
                                else:
                                    recognize_and_check(knn_clf, check_type='in')
                                    print("You have successfully checked in.")
                                break  # Automatically log out after check-in
                            elif employee_choice == '2':
                                if already_checked_out(employee_id):
                                    print("You have already checked out today.")
                                else:
                                    recognize_and_check(knn_clf, check_type='out')
                                    print("You have successfully checked out.")
                                break  # Automatically log out after check-out
                            elif employee_choice == '3':
                                print("Logging out...")
                                break
                            else:
                                print("Invalid choice. Please try again.") 
                                
                                
                            
                
                else:
                    print("Invalid username or password.")
                
                cursor.close()
                conn.close()
            
            elif choice == '3':
                print("Exiting the system. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

    main()

    return render(request, 'face/index.html')
