import cv2
import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

class VisionScreeningSystem:
    def __init__(self):
        print("🔍 Initializing AI Vision Screening System...")
        
        # Load OpenCV Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Create directories
        self.results_dir = "vision_screening_results"
        self.reports_dir = os.path.join(self.results_dir, "reports")
        self.graphs_dir = os.path.join(self.results_dir, "graphs")
        
        for dir_path in [self.results_dir, self.reports_dir, self.graphs_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Test sequences
        self.tests = {
            'distance': {
                'name': 'Distance Vision Test (Myopia Check)',
                'type': 'snellen_chart',
                'letters': ['E', 'H', 'N', 'O', 'S', 'V', 'Z', 'T', 'L', 'C', 'F', 'P', 'D'],
                'sizes': [100, 80, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6],
                'current_index': 0,
            },
            'near': {
                'name': 'Near Vision Test (Hyperopia Check)',
                'type': 'reading_test',
                'texts': [
                    "The quick brown fox jumps over the lazy dog.",
                    "Pack my box with five dozen liquor jugs.",
                    "How vexingly quick daft zebras jump!",
                    "Bright vixens jump; dozy fowl quack.",
                    "Jinxed wizards pluck ivy from the big quilt."
                ],
                'font_sizes': [40, 35, 30, 25, 20],
                'current_index': 0,
            },
            'astigmatism': {
                'name': 'Astigmatism Test',
                'type': 'radial_lines',
                'angles': [0, 90, 45, 135, 0],
                'line_widths': [5, 4, 3, 2, 1],
                'current_index': 0,
            },
            'contrast': {
                'name': 'Contrast Sensitivity Test',
                'type': 'contrast_grid',
                'levels': [100, 80, 60, 40, 20],
                'current_index': 0,
            }
        }
        
        # Eye tracking data
        self.eye_tracking_data = []
        self.test_responses = []
        
        print("✅ Vision Screening System Ready!")

    def detect_eyes(self, frame):
        """Detect eyes and return metrics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=6, 
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        eye_metrics = {
            'detected': False,
            'left_eye': None,
            'right_eye': None,
            'pupil_distance': 0,
            'timestamp': time.time()
        }
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y+h//2, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=8, 
                minSize=(30, 30),
                maxSize=(100, 100)
            )
            
            if len(eyes) >= 2:
                eye_metrics['detected'] = True
                eyes = sorted(eyes, key=lambda e: e[0])
                
                for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                    eye_x = x + ex + ew // 2
                    eye_y = y + ey + eh // 2
                    
                    if i == 0:
                        eye_metrics['left_eye'] = (eye_x, eye_y)
                    else:
                        eye_metrics['right_eye'] = (eye_x, eye_y)
                    
                    cv2.rectangle(frame, 
                                (x + ex, y + ey), 
                                (x + ex + ew, y + ey + eh), 
                                (0, 0, 255), 2)
                    cv2.circle(frame, (eye_x, eye_y), 3, (255, 255, 0), -1)
                    cv2.putText(frame, "L" if i == 0 else "R", 
                              (x + ex, y + ey - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                if eye_metrics['left_eye'] and eye_metrics['right_eye']:
                    dx = eye_metrics['right_eye'][0] - eye_metrics['left_eye'][0]
                    dy = eye_metrics['right_eye'][1] - eye_metrics['left_eye'][1]
                    eye_metrics['pupil_distance'] = np.sqrt(dx**2 + dy**2)
                
                # Store eye tracking data
                self.eye_tracking_data.append({
                    'timestamp': time.time(),
                    'left_eye_x': eye_metrics['left_eye'][0] if eye_metrics['left_eye'] else 0,
                    'left_eye_y': eye_metrics['left_eye'][1] if eye_metrics['left_eye'] else 0,
                    'right_eye_x': eye_metrics['right_eye'][0] if eye_metrics['right_eye'] else 0,
                    'right_eye_y': eye_metrics['right_eye'][1] if eye_metrics['right_eye'] else 0,
                    'pupil_distance': eye_metrics['pupil_distance']
                })
        
        return frame, eye_metrics

    def analyze_results(self, responses):
        """Analyze test results"""
        if not responses:
            return {
                'overall_score': 0,
                'condition': 'No Data',
                'glasses_needed': False,
                'glasses_recommendation': 'Complete tests for analysis',
                'detailed_analysis': {}
            }
        
        self.test_responses = responses
        
        total_weight = 0
        weighted_score = 0
        
        for response in responses:
            difficulty = response.get('difficulty', 1.0)
            user_answer = response.get('user_answer', '').upper()
            
            weight = difficulty
            
            if user_answer == 'Y':
                weighted_score += 10 * weight
            elif user_answer == 'N':
                weighted_score += 3 * weight
            
            total_weight += weight * 10
        
        overall_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0
        
        analysis_by_test = {}
        for test_key in ['distance', 'near', 'astigmatism', 'contrast']:
            test_responses = [r for r in responses if r.get('test_type') == test_key]
            if test_responses:
                correct_count = len([r for r in test_responses if r.get('user_answer', '').upper() == 'Y'])
                test_score = (correct_count / len(test_responses)) * 100
                analysis_by_test[test_key] = {
                    'score': test_score,
                    'total_tests': len(test_responses),
                    'clear_responses': correct_count
                }
        
        # Determine condition
        if overall_score >= 85:
            condition = "Excellent Vision"
            glasses_needed = False
            recommendation = "Your vision is excellent. No glasses needed at this time."
        elif overall_score >= 70:
            condition = "Good Vision"
            glasses_needed = False
            recommendation = "Your vision is good. Regular checkups recommended."
        elif overall_score >= 50:
            condition = "Moderate Vision Issues"
            glasses_needed = True
            recommendation = "Recommended: Prescription glasses for daily use."
        else:
            condition = "Significant Vision Issues"
            glasses_needed = True
            recommendation = "URGENT: Consult eye doctor immediately."
        
        return {
            'overall_score': overall_score,
            'condition': condition,
            'glasses_needed': glasses_needed,
            'glasses_recommendation': recommendation,
            'detailed_analysis': analysis_by_test,
            'test_count': len(responses)
        }

    def generate_graphs(self, analysis_results, patient_name="User"):
        """Generate graphs - NO RADAR CHART"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graphs_data = []
        
        # ---------- GRAPH 1: Eye Movement Trajectory ----------
        fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if self.eye_tracking_data and len(self.eye_tracking_data) > 5:
            data_points = min(1000, len(self.eye_tracking_data))
            
            if data_points > 1:
                start_time = self.eye_tracking_data[0]['timestamp']
                timestamps = [self.eye_tracking_data[i]['timestamp'] - start_time for i in range(data_points)]
                
                # Left eye
                left_eye_y = [self.eye_tracking_data[i]['left_eye_y'] for i in range(data_points)]
                if max(left_eye_y) > min(left_eye_y):
                    left_eye_y = [(y - min(left_eye_y)) / (max(left_eye_y) - min(left_eye_y) + 0.001) * 10 for y in left_eye_y]
                
                # Right eye
                right_eye_y = [self.eye_tracking_data[i]['right_eye_y'] for i in range(data_points)]
                if max(right_eye_y) > min(right_eye_y):
                    right_eye_y = [(y - min(right_eye_y)) / (max(right_eye_y) - min(right_eye_y) + 0.001) * 10 for y in right_eye_y]
                
                # Plot Left Eye
                axes[0].plot(timestamps, left_eye_y, 'r--', linewidth=1.5, label='Left Eye Movement Trajectory')
                axes[0].set_xlabel('time (s)')
                axes[0].set_ylabel('left eye movement (mm)')
                axes[0].set_title('Left Eye Movement Trajectory', fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot Right Eye
                axes[1].plot(timestamps, right_eye_y, 'b--', linewidth=1.5, label='Right Eye Movement Trajectory')
                axes[1].set_xlabel('time (s)')
                axes[1].set_ylabel('right eye movement (mm)')
                axes[1].set_title('Right Eye Movement Trajectory', fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Eye Tracking Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        eye_tracking_path = os.path.join(self.graphs_dir, f"eye_tracking_analysis_{timestamp}.png")
        fig1.savefig(eye_tracking_path, dpi=100, bbox_inches='tight')
        graphs_data.append(("Eye Tracking Analysis", eye_tracking_path))
        
        # ---------- GRAPH 2: Test Performance Bar Chart ----------
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        detailed = analysis_results.get('detailed_analysis', {})
        test_names = []
        test_scores = []
        
        for test_key in ['distance', 'near', 'astigmatism', 'contrast']:
            if test_key in detailed:
                if test_key == 'distance':
                    test_names.append('Distance Vision')
                elif test_key == 'near':
                    test_names.append('Near Vision')
                elif test_key == 'astigmatism':
                    test_names.append('Astigmatism')
                elif test_key == 'contrast':
                    test_names.append('Contrast')
                test_scores.append(detailed[test_key]['score'])
        
        if test_names:
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            bars = ax2.bar(test_names, test_scores, color=colors[:len(test_names)])
            ax2.set_ylabel('Score (%)')
            ax2.set_title('Test-by-Test Performance', fontweight='bold')
            ax2.set_ylim(0, 100)
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, score in zip(bars, test_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        performance_path = os.path.join(self.graphs_dir, f"test_performance_{timestamp}.png")
        fig2.savefig(performance_path, dpi=100, bbox_inches='tight')
        graphs_data.append(("Test Performance", performance_path))
        
        # ---------- GRAPH 3: Glasses Recommendation ----------
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        glasses_needed = analysis_results.get('glasses_needed', False)
        overall_score = analysis_results.get('overall_score', 0)
        
        if glasses_needed:
            labels = ['Glasses Needed', 'Vision OK']
            sizes = [80, 20]
            colors = ['#e74c3c', '#2ecc71']
            explode = (0.1, 0)
            title = f'Glasses Recommendation: NEEDED\nOverall Score: {overall_score:.1f}%'
        else:
            labels = ['Vision OK', 'Monitor']
            sizes = [90, 10]
            colors = ['#2ecc71', '#f39c12']
            explode = (0.1, 0)
            title = f'Glasses Recommendation: NOT NEEDED\nOverall Score: {overall_score:.1f}%'
        
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax3.axis('equal')
        ax3.set_title(title, fontweight='bold')
        
        glasses_path = os.path.join(self.graphs_dir, f"glasses_recommendation_{timestamp}.png")
        fig3.savefig(glasses_path, dpi=100, bbox_inches='tight')
        graphs_data.append(("Glasses Recommendation", glasses_path))
        
        plt.close('all')
        return graphs_data

    def save_report(self, analysis_results, graphs_data, patient_info={}):
        """Save CSV and TXT only"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ---------- CSV FILE ----------
        csv_file = os.path.join(self.reports_dir, f"vision_screening_data_{timestamp}.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(['PATIENT INFORMATION'])
            writer.writerow(['Name', patient_info.get('name', 'Unknown')])
            writer.writerow(['Age Group', patient_info.get('age_group', 'Unknown')])
            writer.writerow(['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([])
            
            writer.writerow(['OVERALL RESULTS'])
            writer.writerow(['Overall Score', f"{analysis_results['overall_score']:.1f}%"])
            writer.writerow(['Vision Condition', analysis_results['condition']])
            writer.writerow(['Glasses Needed', 'YES' if analysis_results['glasses_needed'] else 'NO'])
            writer.writerow([])
            
            writer.writerow(['TEST RESULTS'])
            writer.writerow(['Test Type', 'Score (%)', 'Tests', 'Clear Responses'])
            
            detailed = analysis_results.get('detailed_analysis', {})
            for test_key, data in detailed.items():
                test_name = {
                    'distance': 'Distance Vision',
                    'near': 'Near Vision',
                    'astigmatism': 'Astigmatism',
                    'contrast': 'Contrast'
                }.get(test_key, test_key)
                
                writer.writerow([
                    test_name,
                    f"{data['score']:.1f}",
                    data['total_tests'],
                    data['clear_responses']
                ])
        
        print(f"📊 CSV: {os.path.basename(csv_file)}")
        
        # ---------- TXT FILE ----------
        txt_file = os.path.join(self.reports_dir, f"vision_report_{timestamp}.txt")
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("           VISION SCREENING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Patient: {patient_info.get('name', 'Unknown')}\n")
            f.write(f"Age Group: {patient_info.get('age_group', 'Unknown')}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Overall Score: {analysis_results['overall_score']:.1f}%\n")
            f.write(f"Condition: {analysis_results['condition']}\n")
            f.write(f"Glasses Needed: {'YES' if analysis_results['glasses_needed'] else 'NO'}\n\n")
            
            f.write("Test Results:\n")
            f.write("-" * 50 + "\n")
            
            for test_key, data in detailed.items():
                test_name = {
                    'distance': 'Distance Vision',
                    'near': 'Near Vision',
                    'astigmatism': 'Astigmatism',
                    'contrast': 'Contrast'
                }.get(test_key, test_key)
                
                f.write(f"\n{test_name}: {data['score']:.1f}%\n")
                f.write(f"  Tests: {data['total_tests']}, Clear: {data['clear_responses']}\n")
            
            f.write(f"\nRecommendation: {analysis_results['glasses_recommendation']}\n\n")
            f.write("=" * 70 + "\n")
        
        print(f"📄 TXT: {os.path.basename(txt_file)}")
        print(f"📈 Graphs: {len(graphs_data)} files generated")
        
        return txt_file, graphs_data

    def analyze_eye_behavior(self):
        """Simple eye behavior analysis"""
        if len(self.eye_tracking_data) < 10:
            return {'status': 'Insufficient data'}
        
        return {
            'status': 'Analysis complete',
            'data_points': len(self.eye_tracking_data)
        }