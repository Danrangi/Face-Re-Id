import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import tempfile

# Import custom modules
import config
from utils.database import DatabaseManager
from utils.face_detection import FaceDetector
from utils.face_recognition import FaceRecognizer

# Initialize components
@st.cache_resource
def init_components():
    return {
        'db': DatabaseManager(),
        'detector': FaceDetector(),
        'recognizer': FaceRecognizer()
    }

def validate_student_info(matric_number, name, department):
    """Validate student information"""
    import re
    
    errors = []
    
    if len(name) < config.MIN_NAME_LENGTH:
        errors.append(f"Name must be at least {config.MIN_NAME_LENGTH} characters")
    
    if not re.match(config.MATRIC_NUMBER_PATTERN, matric_number):
        errors.append("Invalid matric number format")
    
    if not department or len(department.strip()) == 0:
        errors.append("Department is required")
    
    return errors

def main():
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="🎓",
        layout="wide"
    )
    
    st.title(config.APP_TITLE)
    
    # Initialize components
    components = init_components()
    db = components['db']
    detector = components['detector']
    recognizer = components['recognizer']
    
    # Sidebar navigation
    st.sidebar.title(config.SIDEBAR_TITLE)
    page = st.sidebar.radio(
        "Select Function",
        ["📝 Register Student", "✅ Mark Attendance", "📊 View Records", "ℹ️ About"]
    )
    
    # Page: Register Student
    if page == "📝 Register Student":
        st.header("Student Registration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Student Information")
            matric_number = st.text_input("Matric Number", placeholder="e.g., CS2021001").upper()
            name = st.text_input("Full Name", placeholder="e.g., John Doe")
            department = st.text_input("Department", placeholder="e.g., Computer Science")
        
        with col2:
            st.subheader("Capture Face")
            capture_method = st.radio("Choose method:", ["📷 Use Webcam", "📤 Upload Image"])
            
            face_image = None
            
            if capture_method == "📷 Use Webcam":
                camera_input = st.camera_input("Take a photo")
                if camera_input:
                    face_image = Image.open(camera_input)
            else:
                uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
                if uploaded_file:
                    face_image = Image.open(uploaded_file)
            
            if face_image:
                st.image(face_image, caption="Captured Image", use_column_width=True)
        
        # Registration button
        if st.button("Register Student", type="primary"):
            # Validate inputs
            errors = validate_student_info(matric_number, name, department)
            
            if errors:
                for error in errors:
                    st.error(error)
            elif face_image is None:
                st.error("Please capture or upload a face image")
            else:
                with st.spinner("Processing registration..."):
                    # Detect face
                    cropped_face, bbox = detector.detect_face(face_image)
                    
                    if cropped_face is None:
                        st.error("No face detected in the image. Please try again.")
                    else:
                        # Extract embedding
                        embedding = recognizer.get_embedding(cropped_face)
                        
                        # Save to database
                        success, message = db.add_student(
                            matric_number, name, department, embedding
                        )
                        
                        if success:
                            st.success(message)
                            st.balloons()
                        else:
                            st.error(message)
    
    # Page: Mark Attendance
    elif page == "✅ Mark Attendance":
        st.header("Mark Attendance")
        
        # Load registered students
        students_df = db.get_all_students()
        
        if students_df.empty:
            st.warning("No students registered yet. Please register students first.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Capture Face for Attendance")
                capture_method = st.radio("Choose method:", ["📷 Use Webcam", "📤 Upload Image"])
                
                face_image = None
                
                if capture_method == "📷 Use Webcam":
                    camera_input = st.camera_input("Take a photo for attendance")
                    if camera_input:
                        face_image = Image.open(camera_input)
                else:
                    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
                    if uploaded_file:
                        face_image = Image.open(uploaded_file)
                
                if face_image:
                    st.image(face_image, caption="Captured Image", use_column_width=True)
            
            with col2:
                st.subheader("Recognition Result")
                
                if st.button("Process Attendance", type="primary") and face_image:
                    with st.spinner("Recognizing face..."):
                        # Detect face
                        cropped_face, bbox = detector.detect_face(face_image)
                        
                        if cropped_face is None:
                            st.error("No face detected. Please try again.")
                        else:
                            # Extract embedding
                            query_embedding = recognizer.get_embedding(cropped_face)
                            
                            # Prepare database embeddings
                            db_embeddings = []
                            for _, student in students_df.iterrows():
                                if isinstance(student['embedding'], list):
                                    db_embeddings.append((
                                        student['matric_number'],
                                        student['name'],
                                        student['department'],
                                        np.array(student['embedding'])
                                    ))
                            
                            # Find match
                            match = recognizer.find_match(query_embedding, db_embeddings)
                            
                            if match:
                                st.success(f"✅ Student Identified!")
                                st.write(f"**Name:** {match['name']}")
                                st.write(f"**Matric Number:** {match['matric_number']}")
                                st.write(f"**Department:** {match['department']}")
                                st.write(f"**Confidence:** {match['confidence']}")
                                
                                # Mark attendance
                                success, message = db.mark_attendance(
                                    match['matric_number'],
                                    match['name'],
                                    match['department']
                                )
                                
                                if success:
                                    st.success(message)
                                    st.balloons()
                                else:
                                    st.warning(message)
                            else:
                                st.error("❌ No match found. Student may not be registered.")
    
    # Page: View Records
    elif page == "📊 View Records":
        st.header("Attendance Records")
        
        tab1, tab2 = st.tabs(["📋 Attendance History", "👥 Registered Students"])
        
        with tab1:
            # Get initial data to check departments
            attendance_df = db.get_attendance_records()
            
            # Filters
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                filter_date = st.date_input("Filter by date", value=None)
            
            with col2:
                if not attendance_df.empty and 'department' in attendance_df.columns:
                    departments = attendance_df['department'].unique()
                    filter_dept = st.selectbox("Filter by department", ["All"] + list(departments))
                else:
                    filter_dept = st.selectbox("Filter by department", ["All"])
            
            with col3:
                if st.button("🔄 Refresh Data"):
                    st.rerun()
            
            # Get filtered records
            attendance_df = db.get_attendance_records(
                filter_date=filter_date.strftime(config.DATE_FORMAT) if filter_date else None,
                filter_department=None if filter_dept == "All" else filter_dept
            )
            
            if attendance_df.empty:
                st.info("No attendance records found.")
            else:
                st.dataframe(attendance_df, use_container_width=True)
                
                # Export option
                csv = attendance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with tab2:
            students_df = db.get_all_students()
            
            if students_df.empty:
                st.info("No students registered yet.")
            else:
                # Don't show embeddings in display
                display_df = students_df.drop(columns=['embedding'])
                st.dataframe(display_df, use_container_width=True)
                st.write(f"Total registered students: {len(students_df)}")
    
    # Page: About
    else:
        st.header("About This System")
        
        st.markdown("""
        ### 🎓 Face-Based Attendance System
        
        This smart attendance system uses advanced face recognition technology to:
        
        - **Automatically identify students** using facial features
        - **Mark attendance in real-time** without manual intervention
        - **Prevent proxy attendance** through biometric verification
        - **Generate detailed reports** for record keeping
        
        #### 🔧 Technical Details:
        - **Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks)
        - **Face Recognition:** FaceNet with Inception ResNet V1
        - **Similarity Metric:** Cosine Similarity
        - **Current Threshold:** {:.1f}
        
        #### 📊 System Statistics:
        """.format(config.FACE_RECOGNITION_THRESHOLD))
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_students = len(db.get_all_students())
            st.metric("Registered Students", total_students)
        
        with col2:
            total_attendance = len(db.get_attendance_records())
            st.metric("Total Attendance Records", total_attendance)

if __name__ == "__main__":
    main()
