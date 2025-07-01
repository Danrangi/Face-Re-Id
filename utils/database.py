import pandas as pd
import os
import json
from datetime import datetime
import config

class DatabaseManager:
    def __init__(self):
        self.students_path = config.STUDENTS_DB_PATH
        self.attendance_path = config.ATTENDANCE_DB_PATH
        self._initialize_databases()
    
    def _initialize_databases(self):
        """Create CSV files if they don't exist"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.students_path), exist_ok=True)
        
        # Students database
        if not os.path.exists(self.students_path):
            students_df = pd.DataFrame(columns=['matric_number', 'name', 'department', 'embedding', 'registered_date'])
            students_df.to_csv(self.students_path, index=False)
        
        # Attendance database
        if not os.path.exists(self.attendance_path):
            attendance_df = pd.DataFrame(columns=['matric_number', 'name', 'department', 'date', 'time', 'status'])
            attendance_df.to_csv(self.attendance_path, index=False)
    
    def add_student(self, matric_number, name, department, embedding):
        """Add a new student to the database"""
        try:
            # Read existing data or create empty dataframe
            if os.path.exists(self.students_path) and os.path.getsize(self.students_path) > 0:
                students_df = pd.read_csv(self.students_path)
            else:
                students_df = pd.DataFrame(columns=['matric_number', 'name', 'department', 'embedding', 'registered_date'])
            
            # Check if student already exists
            if matric_number in students_df['matric_number'].values:
                return False, "Student with this matric number already exists"
            
            # Convert embedding to JSON string for storage
            embedding_str = json.dumps(embedding.tolist())
            
            # Add new student
            new_student = pd.DataFrame([{
                'matric_number': matric_number,
                'name': name,
                'department': department,
                'embedding': embedding_str,
                'registered_date': datetime.now().strftime(config.TIME_FORMAT)
            }])
            
            students_df = pd.concat([students_df, new_student], ignore_index=True)
            students_df.to_csv(self.students_path, index=False)
            
            return True, "Student registered successfully"
        except Exception as e:
            return False, f"Error registering student: {str(e)}"
    
    def get_all_students(self):
        """Get all registered students with their embeddings"""
        try:
            if not os.path.exists(self.students_path) or os.path.getsize(self.students_path) == 0:
                return pd.DataFrame(columns=['matric_number', 'name', 'department', 'embedding', 'registered_date'])
            
            students_df = pd.read_csv(self.students_path)
            
            # Parse embeddings from JSON strings
            for idx, row in students_df.iterrows():
                if pd.notna(row['embedding']):
                    try:
                        students_df.at[idx, 'embedding'] = json.loads(row['embedding'])
                    except:
                        students_df.at[idx, 'embedding'] = []
            
            return students_df
        except Exception as e:
            print(f"Error reading students database: {str(e)}")
            return pd.DataFrame(columns=['matric_number', 'name', 'department', 'embedding', 'registered_date'])
    
    def mark_attendance(self, matric_number, name, department):
        """Mark attendance for a student"""
        try:
            # Read existing data or create empty dataframe
            if os.path.exists(self.attendance_path) and os.path.getsize(self.attendance_path) > 0:
                attendance_df = pd.read_csv(self.attendance_path)
            else:
                attendance_df = pd.DataFrame(columns=['matric_number', 'name', 'department', 'date', 'time', 'status'])
            
            # Check if already marked today
            today = datetime.now().strftime(config.DATE_FORMAT)
            today_attendance = attendance_df[
                (attendance_df['matric_number'] == matric_number) & 
                (attendance_df['date'] == today)
            ]
            
            if not today_attendance.empty:
                return False, "Attendance already marked for today"
            
            # Add attendance record
            new_record = pd.DataFrame([{
                'matric_number': matric_number,
                'name': name,
                'department': department,
                'date': today,
                'time': datetime.now().strftime("%H:%M:%S"),
                'status': 'Present'
            }])
            
            attendance_df = pd.concat([attendance_df, new_record], ignore_index=True)
            attendance_df.to_csv(self.attendance_path, index=False)
            
            return True, "Attendance marked successfully"
        except Exception as e:
            return False, f"Error marking attendance: {str(e)}"
    
    def get_attendance_records(self, filter_date=None, filter_department=None):
        """Get attendance records with optional filters"""
        try:
            if not os.path.exists(self.attendance_path) or os.path.getsize(self.attendance_path) == 0:
                return pd.DataFrame(columns=['matric_number', 'name', 'department', 'date', 'time', 'status'])
            
            attendance_df = pd.read_csv(self.attendance_path)
            
            if filter_date:
                attendance_df = attendance_df[attendance_df['date'] == filter_date]
            
            if filter_department:
                attendance_df = attendance_df[attendance_df['department'] == filter_department]
            
            # Sort by date and time if not empty
            if not attendance_df.empty:
                attendance_df = attendance_df.sort_values(by=['date', 'time'], ascending=[False, False])
            
            return attendance_df
        except Exception as e:
            print(f"Error reading attendance database: {str(e)}")
            return pd.DataFrame(columns=['matric_number', 'name', 'department', 'date', 'time', 'status'])
